module wfmod

using LinearAlgebra

function normalize_wf!(wf)
    # Normalize wavefunction
    normwf = norm(wf)
    wf./=normwf
    return wf
end

function expectation_value_normalized(wf,oper,tempwf)
    # Expectation value 
    # Calculates <psi|oper|psi> / <psi|psi>
    mul!(tempwf,oper,wf)
    return dot(wf,tempwf)/(norm(wf)^2)
end

function expectation_value(wf,oper,tempwf)
    # Expectation value 
    # Calculates <psi|oper|psi> 
    mul!(tempwf,oper,wf)
    return dot(wf,tempwf)
end

function wf_braket(wf1,wf2,oper,tempwf)
    # Calculates <psi1|oper|psi2> 
    mul!(tempwf,oper,wf2)
    return dot(wf1,tempwf)
end

function wf_braket(wf1,wf2,oper,tempwf)
    # Calculates <psi1|oper|psi2> 
    mul!(tempwf,oper,wf2)
    return dot(wf1,tempwf)
end

function wf_operdag_oper_wf(wf,oper,tempwf)
    # Calculates <psi|oper^dag oper |psi> 
    mul!(tempwf,oper,wf)
    return norm(tempwf)^2
end

function wf_collapse!(wf,oper,tempwf)
    mul!(tempwf,oper,wf)
    wf.=tempwf
    normalize_wf!(wf)
    return wf
end
    
end