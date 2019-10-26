build_number=4
kubectl set image deployment/cicdpipeline-deployment cicdpipeline=kmitsunami/cicdpipeline:$build_number
