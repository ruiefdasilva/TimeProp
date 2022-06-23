module TimeProp
# TimeProp is a module to solve the TDSE and the Lindblad master equation
# In principle it will have both MCWF approach and full master equation aproach

# Imports of Julia
println("Loading TimeProp module")
println("Number of Julia threads: ", Threads.nthreads())
using LinearAlgebra; println("LinearAlgebra loaded"); println("Number of BLAS threads: ", BLAS.get_num_threads())
if Threads.nthreads()>=1
    println("Changing the number of BLAS threads to be the same as Julia threads!")
    BLAS.set_num_threads(Threads.nthreads())
end
using SparseArrays; println("SparseArrays loaded")
using DifferentialEquations ; println("DifferentialEquations loaded")
using PyCall; println("PyCall loaded")
using Random; println("Random loaded")
sp_sparse=pyimport("scipy.sparse") ; println("scipy.sparse loaded")
try
    using MKLSparse; println("Using MKLSparse")
catch
    println("Not using MKLSparse")
end
try 
    using CUDA, CUDA.CUSPARSE, CUDA.CUBLAS; println("CUDA is available")
    CUDA.allowscalar(false) ; println("CUDA: scalar indexing is forbidden, as it should be in a proper GPU calculation.")
    cuda_avail = true
catch
    println("CUDA is not available")
    cuda_avail = false
end
include("./wfmod.jl"); using .wfmod

using PyCall
pushfirst!(PyVector(pyimport("sys")."path"), "")
iofiles=pyimport("iofiles");

function pysparse_to_julia(A)
    A = sp_sparse.csc_matrix(A)
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = A.data
    SparseMatrixCSC{eltype(nzVal),Int}(m, n, colPtr, rowVal, nzVal)
end

function qobj_to_julia(A;density_matrix=false)
    if A.type=="oper"
        if density_matrix
            return A.full()
        else
            return pysparse_to_julia(A.data)
        end
    elseif A.type=="ket"
        if density_matrix
            tmp = A.full()*1.0
            return tmp*tmp'
        else
            return A.full()
        end
    end
end

function scale_matrix!(A,alpha)
    N=size(A)[1]
    M=size(A)[2]
    Threads.@threads for j in 1:M
        for i in 1:N
            A[i,j]*=alpha
        end    
    end
    return A
end




function adjoint_time_prop!(A,B)
    N=size(B)[1]
    M=size(B)[2]
    Threads.@threads for i in 1:N
        for j in 1:M
            A[j,i]=conj(B[i,j])
        end    
    end
    return A
end

function sesolve(H0,H0_td,fs_H0_td,psi0,ts,e_ops;dt=nothing,rtol=1e-10,atol=1e-10)
    # H0 is the time independent Hamiltonian
    # H0_td is a list of a operators such as 
    # H(t) = H0 + sum_i H0_td[i] fs_H0_td[i](t)
    # psi0 is the initial wavefunction
    # ts is an array of times for the propagation
    # e_ops is a list of operators to perform expectation values
    println("-"^100); println("Starting a time-dependent Schrodinger equation")
    temp = psi0 .* 0.0
    results = zeros(ComplexF64,length(ts),length(e_ops))
    ps = [H0,H0_td,fs_H0_td,temp]
    if dt==nothing
        dt=ts[2]-ts[1]
    end
    
    function minus_im_Heff_sesolve!(psires,psi,ps,t)
        H0,H0_td,fs_H0_td,temp = ps
        # Hamiltonian part
        mul!(psires,H0,psi,-1.0im,0.0)
        for i in 1:length(H0_td)
            mul!(psires,H0_td[i],psi,-1.0im*fs_H0_td[i](t),1.0)
        end
        return psires
    end 
    
    function get_callback(ts,results,e_ops,temp)
        function savefunc(u,t,integrator)
            it=findall(ts.≈t)[1]
            for i_eop in 1:length(e_ops)
                results[it,i_eop] = wfmod.expectation_value(u,e_ops[i_eop],temp)
            end
            return nothing
        end
        FunctionCallingCallback(savefunc; funcat=ts)
    end
    cb2=get_callback(ts,results,e_ops,temp)
    cb=CallbackSet([cb2]...);
    prob = ODEProblem(minus_im_Heff_sesolve!,psi0,(ts[1],ts[end]),ps)
    sol = solve(prob,Tsit5(),save_start=false,save_end=false,save_everystep=false,callback=cb,reltol=rtol,abstol=atol,maxiters=1e6,dt=dt)
    
    tempwf = nothing
    println("-"^100)
    return results
end


function mcsolve(H0,H0_td,fs_H0_td,c_ops_no_td,c_ops_td,fs_c_ops_td,psi0,ts,e_ops;ntrajs = 1, dt=nothing,rtol=1e-10,atol=1e-10,verbose = false, seed_number = nothing , print_intermidiate = false, ntrajs_intermidiate = 10 , name_files_intermidiate = nothing)
    # MCWF solver for a single psi0. If you want to propagate an initial state that is
    # rho0 = \sum_i p_i psi_i * psi_i ^dag you just need to run this wavefunction and sum all results * p_i
    # H0 is the time independent Hamiltonian
    # H0_td is a list of a operators such as 
    # H(t) = H0 + sum_i H0_td[i] fs_H0_td[i](t)
    # c_ops_no_td is a list of collapse operators
    # c_ops_td is a list of collapse operators is a list of collapse operators with time-dependency
    # psi0 is the initial wavefunction
    # ts is an array of times for the propagation
    # e_ops is a list of operators to perform expectation values
    println("-"^100); println("Starting a time-dependent MCWF Lindblad equation")
    if seed_number!=nothing
        println("Using a seed for random numbers, seed: ", seed_number)
        
        Random.seed!(seed_number)
    end
    results = zeros(ComplexF64,length(ts),length(e_ops))
    results_all = results.*0.0
    temp = psi0 .* 0.0
    deltap = rand()

    ps=(H0,H0_td,fs_H0_td,c_ops_no_td,c_ops_td,fs_c_ops_td,temp)
    if dt==nothing
        dt=ts[2]-ts[1]
    end
    
    function minus_im_Heff_mcsolve!(psires,psi,ps,t)
        # ttt=time()
        H0,H0_td,fs_H0_td,c_ops_no_td,c_ops_td,fs_c_ops_td,temp = ps
        # Hamiltonian part
        mul!(psires,H0,psi,-1.0im,0.0)
        for i in 1:length(H0_td)
            mul!(psires,H0_td[i],psi,-1.0im*fs_H0_td[i](t),1.0)
        end
        
        for i in 1:length(c_ops_no_td)
            mul!(temp,c_ops_no_td[i],psi,-1.0im,0.0)
            mul!(psires,c_ops_no_td[i]',temp,-0.5im,1.0)
        end
        
        for i in 1:length(c_ops_td)
            mul!(temp,c_ops_td[i],psi,-1.0im*fs_c_ops_td[i](t),0.0)
            mul!(psires,c_ops_td[i]',temp,-0.5im * conj(fs_c_ops_td[i](t)),1.0)
        end
        # println("Calling minus_im_Heff_mcsolve took: ", time()-ttt)
        return psires
    end
    
    function condition(u,t,integrator)
        #ttt=time()
        #println(norm(u)^2, " ", deltap, " ",t)
        result = norm(u)^2 -deltap
        #println("Calling condition took: ", time()-ttt)
        return result
    end

    function affect!(integrator)
        #ttt=time()
        #println(integrator.t , " ",deltap )
        #println(integrator.t , " ",deltap )
        n_cops = length(c_ops_td)+length(c_ops_no_td)
        probabilities = zeros(n_cops)
        for i in 1:length(c_ops_td)
            mul!(temp , c_ops_td[i],integrator.u,fs_c_ops_td[i](integrator.t),0.0)
            probabilities[i] = norm(temp)^2
        end
        for i in 1:length(c_ops_no_td)
            mul!(temp , c_ops_no_td[i],integrator.u,1.0,0.0)
            probabilities[i+length(c_ops_td)] = norm(temp)^2
        end
        
        for i in 2:n_cops
            probabilities[i] = probabilities[i-1] + probabilities[i]
        end
        
        rand_number = rand() * probabilities[end]
        
        
        i_op=0
        for i in 1:n_cops
            if rand_number <=probabilities[i]
                i_op = i*1
                break
            end
        end
        
        if 1<=i_op <=length(c_ops_td)
            wfmod.wf_collapse!(integrator.u , c_ops_td[i_op],temp)
        end
        
        if i_op>length(c_ops_td)
            wfmod.wf_collapse!(integrator.u , c_ops_no_td[i_op-length(c_ops_td)],temp)
        end

        #println("Calling affect took: ", time()-ttt)
        deltap = rand()
    end    
    
    function get_callback(ts,results,e_ops,temp)
        function savefunc(u,t,integrator)
            #ttt=time()
            it=findall(ts.≈t)[1]
            for i_eop in 1:length(e_ops)
                results[it,i_eop] = wfmod.expectation_value_normalized(u,e_ops[i_eop],temp)
            end
            #println("Calling savefunc took: ", time()-ttt)
            return nothing
        end
        FunctionCallingCallback(savefunc; funcat=ts)
    end

    
    cb1=ContinuousCallback(condition,affect!,save_positions=(false,false))
    cb2=get_callback(ts,results,e_ops,temp)
    cb=CallbackSet([cb1,cb2]...);
    prob = ODEProblem(minus_im_Heff_mcsolve!,psi0,(ts[1],ts[end]),ps)
    for itraj in 1:ntrajs
        t1=time()
        deltap = rand()
        t2=time()
        sol = solve(prob,Tsit5(),save_start=false,save_end=false,save_everystep=false,
                         callback=cb,reltol=rtol,abstol=atol,maxiters=1e18, dt=dt)
        results_all .+=results
        results.*=0.0
        if verbose
            println("Performing : ", itraj, " , out of ntrajs: ", ntrajs , "  Time for traj  : ", time()-t1, " time for setup : ",time()-t2)
            flush(stdout)
        end
        if print_intermidiate
            if itraj==1 || mod(itraj,ntrajs_intermidiate)==0
                for i_eop in 1:length(e_ops)
                    iofiles.WriteFile([ts,real.(results_all[:,i_eop].*(1.0/itraj))]," ",name_files_intermidiate[i_eop])
                end
            end
        end
    end
    tempwf = nothing
    println("-"^100)
    return results_all.*(1.0/ntrajs)
end




function commutator!(result::Array , operator, rho , temp , rhodag)
    # returns   (O * rho - rho * O)
    mul!(temp , operator' , rhodag , -1.0 , 0.0 )
    adjoint_time_prop!(result,temp)
    mul!(result , operator,rho , 1.0 ,1.0)
    return result
end

function commutator!(result::CuArray , operator, rho , temp , rhodag)
    # returns   (O * rho - rho * O)
    # temp = O' * rho' 
    CUSPARSE.mm!('C','N',1.0,operator,rhodag,0.0,temp,'0')
    # result = - rho * O 
    CUBLAS.geam!('N','C',0.0,result,-1.0,temp,result)
    # result = - rho * O  + O * rho
    CUSPARSE.mm!('N','N',1.0,operator,rho,1.0,result,'0')
    return result
end

function dissipator!(result::Array , operator , rho , temp , temp2 )
    # returns  O * rho * O' - 0.5 * (O'*O*rho + rho*O'*O)

    # temp = O * rho
    mul!(temp , operator , rho , 1.0 , 0.0 )
    # result = rho' * O'
    adjoint_time_prop!(result,temp)
    # temp = O * rho' * O'
    mul!(temp, operator , result , 1.0 , 0.0 )
    # result = (O * rho' * O') = O' * rho * O
    adjoint_time_prop!(result,temp)

    # temp = O * rho
    mul!(temp , operator , rho , 1.0, 0.0)
    # result = O' * rho * O - 0.5 * O' * O * rho 
    mul!(result , operator' , temp , -0.5, 1.0)

    # temp = rho'
    adjoint_time_prop!(temp,rho)
    # temp2 = O * rho'
    mul!(temp2, operator ,temp, 1.0 ,0.0)
    # temp = O' * O * rho'
    mul!(temp, operator' , temp2,1.0, 0.0 )
    # temp2 =  rho * O' * O 
    adjoint_time_prop!(temp2,temp)
    # result = O' * rho * O - 0.5 * O' * O * rho  - 0.5 * rho * O' * O 
    BLAS.axpy!( -0.5+0.0im ,temp2 , result )
    return result
end

function dissipator!(result::CuArray , operator , rho , temp , temp2 )
    # returns  O * rho * O' - 0.5 * (O'*O*rho + rho*O'*O)

    # temp = O * rho
    CUSPARSE.mm!('N','N',1.0,operator,rho,0.0,temp,'0')
    # result = rho' * O'
    CUBLAS.geam!('N','C',0.0,result,1.0,temp,result)
    # temp = O * rho' * O'
    CUSPARSE.mm!('N','N',1.0,operator,result,0.0,temp,'0')
    # result = (O * rho' * O') = O' * rho * O
    CUBLAS.geam!('N','C',0.0,result,1.0,temp,result)


    # temp = O * rho
    CUSPARSE.mm!('N','N',1.0,operator,rho,0.0,temp,'0')
    # result = O' * rho * O - 0.5 * O' * O * rho 
    CUSPARSE.mm!('C','N',-0.5,operator,temp,1.0,result,'0')
    

    # temp = rho'
    CUBLAS.geam!('N','C',0.0,temp,1.0,rho,temp)
    # temp2 = O * rho'
    CUSPARSE.mm!('N','N',1.0,operator,temp,0.0,temp2,'0')
    # temp = O' * O * rho'
    CUSPARSE.mm!('C','N',1.0,operator,temp2,0.0,temp,'0')
    # temp2 =  rho * O' * O 
    CUBLAS.geam!('N','C',0.0,temp2,1.0,temp,temp2)
    # result = O' * rho * O - 0.5 * O' * O * rho  - 0.5 * rho * O' * O 
    CUBLAS.geam!('N','N',1.0,result,-0.5,temp,result)
    return result
end










function lindblad!(Lrho::Array,rho,ps,t)
    H0,H0_td,fs_H0_td,c_ops_no_td,c_ops_td,fs_c_ops_td,temp,rhodag,temp3 = ps
    # Form the adjoint of rho in temp2
    adjoint_time_prop!(rhodag,rho)
    # Lrho = - i * [H0 , rho]
    commutator!(Lrho, H0 , rho , temp3 , rhodag)
    scale_matrix!(Lrho,-1.0im)

    for i in 1:length(H0_td)
        commutator!(temp, H0_td[i] , rho , temp3 , rhodag)
        BLAS.axpy!(-1im * fs_H0_td[i](t) ,temp , Lrho )
    end

    for i in 1:length(c_ops_no_td)
        dissipator!(temp , c_ops_no_td[i] , rho , temp3 ,rhodag)
        BLAS.axpy!(1.0+0.0im ,temp , Lrho )
    end
    
    for i in 1:length(c_ops_td)
        dissipator!(temp , c_ops_td[i] , rho , temp3 ,rhodag)
        BLAS.axpy!((1.0+0.0im) *fs_c_ops_td[i](t) * conj(fs_c_ops_td[i](t))  ,temp , Lrho )
    end    
    Lrho
end

function lindblad!(Lrho::CuArray,rho,ps,t)
    H0,H0_td,fs_H0_td,c_ops_no_td,c_ops_td,fs_c_ops_td,temp,rhodag,temp3 = ps
    # Form the adjoint of rho in temp2
    CUBLAS.geam!('N','C',0.0,rhodag,1.0,rho,rhodag)
    # Lrho = - i * [H0 , rho]
    commutator!(Lrho, H0 , rho , temp3 , rhodag)
    Lrho.*=-1.0im

    for i in 1:length(H0_td)
        commutator!(temp, H0_td[i] , rho , temp3 , rhodag)
        CUBLAS.geam!('N','N',1.0,Lrho,-1im * fs_H0_td[i](t),temp,Lrho)
    end

    for i in 1:length(c_ops_no_td)
        dissipator!(temp , c_ops_no_td[i] , rho , temp3 ,rhodag)
        CUBLAS.geam!('N','N',1.0,Lrho,1.0,temp,Lrho)
    end
    
    for i in 1:length(c_ops_td)
        dissipator!(temp , c_ops_td[i] , rho , temp3 ,rhodag)
        CUBLAS.geam!('N','N',1.0,Lrho,(1.0+0.0im) *fs_c_ops_td[i](t) * conj(fs_c_ops_td[i](t)) ,temp,Lrho)
    end    
    Lrho
end


function mesolve(H0,H0_td,fs_H0_td,c_ops_no_td,c_ops_td,fs_c_ops_td,psi0,ts,e_ops; dt=nothing,rtol=1e-10,atol=1e-10,verbose = false)
    # MESOLVE is a master equation solver
    # H0 is the time independent Hamiltonian
    # H0_td is a list of a operators such as 
    # H(t) = H0 + sum_i H0_td[i] fs_H0_td[i](t)
    # c_ops_no_td is a list of collapse operators
    # c_ops_td is a list of collapse operators is a list of collapse operators with time-dependency
    # psi0 is the initial wavefunction
    # ts is an array of times for the propagation
    # e_ops is a list of operators to perform expectation values
    println("-"^100); println("Starting a time-dependent MESOLVE Lindblad equation")
    ttt = time()
    results = zeros(ComplexF64,length(ts),length(e_ops))
    results_all = results.*0.0
    temp = psi0 .* 0.0
    temp2 = psi0 .* 0.0 
    temp3 = psi0 .*0.0
    ps=[H0,H0_td,fs_H0_td,c_ops_no_td,c_ops_td,fs_c_ops_td,temp,temp2,temp3]
    if dt==nothing
        dt=ts[2]-ts[1]
    end
    
    
    
    
    
    function get_callback(ts,results,e_ops,temp)
        function savefunc(u,t,integrator)
            
            it=findall(ts.≈t)[1]
            if verbose
                println("Progress it : ",it," out of ",length(ts), " : ",it*100.0/length(ts) , "%, time since start: " , time()-ttt)
                flush(stdout)
            end
            for i_eop in 1:length(e_ops)
                mul!(temp,e_ops[i_eop],u)
                results[it,i_eop] = tr(temp)
            end
            return nothing
        end
        FunctionCallingCallback(savefunc; funcat=ts)
    end

    
    cb2=get_callback(ts,results,e_ops,temp)
    cb=CallbackSet([cb2]...);
    prob = ODEProblem(lindblad!,psi0,(ts[1],ts[end]),ps)
    sol = solve(prob,Tsit5(),save_start=false,save_end=false,save_everystep=false,callback=cb,reltol=rtol,abstol=atol,maxiters=1e6,dt=dt)
    
    tempwf = nothing
    println("-"^100)
    return results
end

println("TimeProp module loaded!")
end




#=
function commutator_old!(result::Array , operator, rho , temp , rhodag)
    # returns  O * rho - rho * O
    mul!(temp , operator' , rhodag , -1.0 , 0.0 )
    result.=temp'
    mul!(result , operator,rho , 1.0 ,1.0)
    return result
end

function dissipator_old!(result::Array , operator , rho , temp , temp2)
    # returns  O * rho - rho * O
    mul!(temp , operator , rho , 1.0 , 0.0 )
    mul!(result, temp , operator' , 1.0 , 0.0 )

    mul!(temp , operator , rho , 1.0, 0.0)
    mul!(result , operator' , temp , -0.5, 1.0)

    mul!(temp, rho , operator' , 1.0 ,0.0)
    mul!(result, temp , operator,-0.5, 1.0 )
    return result
end
=#

