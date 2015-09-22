function __kubectl_get
  kubectl get $argv[1] --template='{{range .items}}{{.metadata.name}}
{{end}}'
end

function __kubelet_containers_for_pod
  kubectl get pods $argv[1] --template='{{range .status.containerStatuses}}{{.name}}
{{end}}'
end

function __kubectl_clusters
  kubectl config view --output=template --template='{{range .clusters}}{{.name}}
{{end}}'
end

function __kubectl_users
  kubectl config view --output=template --template='{{range .users}}{{.name}}
{{end}}'
end

function __kubectl_contexts
  kubectl config view --output=template --template='{{range .contexts}}{{.name}}
{{end}}'
end

function __kubectl_containers
    set cmd (commandline -opc | grep '^[^-]')
    if test (count $cmd) -eq 3 
        __kubelet_containers_for_pod $cmd[3]
    end
end

function __kubectl_needs_cmd
    set cmd (commandline -opc)
    if test (count $cmd) -eq 1 
        return 0
    end
    return 1
end

function __kubectl_needs_subcmd
    set cmd (commandline -opc | grep '^[^-]')
    if test (count $cmd) -eq 2 
        if __kubectl_using_cmd $argv[1]
            return 0
        end
    end
    return 1
end

function __kubectl_needs_subsubcmd
    set cmd (commandline -opc | grep '^[^-]')
    if test (count $cmd) -eq 3
        if test $argv[2] = '__whatever'
          if __kubectl_using_cmd $argv[1]
            return 0
          end
        end

        if __kubectl_using_subcmd $argv[1] $argv[2]
            return 0
        end
    end
    return 1
end

function __kubectl_using_cmd
    set cmd (commandline -opc | grep '^[^-]')
    if test (count $cmd) -gt 1 
        if test $argv[1] = $cmd[2] 
            return 0
        end
    end
    return 1
end

function __kubectl_using_subcmd
    set cmd (commandline -opc | grep '^[^-]')
    if test (count $cmd) -gt 2 
        if __kubectl_using_cmd $argv[1]
            if test $argv[2] = $cmd[3]
                return 0
            end
        end
    end
    return 1
end


set RESOURCE_COMMANDS get describe patch delete stop label
set RESOURCES         componentStatuses cs endpoints ep events ev limits namespaces \
                      nodes no persistentVolumeClaims pvc persistentVolumes pv pods \
                      pod po podTemplates quota replicationcontrollers rc secrets \
                      serviceAccounts services

for resource_command in $RESOURCE_COMMANDS
    complete -c kubectl -n "__kubectl_needs_subcmd $resource_command" -xA -a '$RESOURCES'
    for resource in $RESOURCES
      complete -c kubectl \
               -n "__kubectl_needs_subsubcmd $resource_command $resource" -xA -a "(__kubectl_get $resource)" 
    end
end

# ===================================================================================
# kubectl *
# ===================================================================================

complete -c kubectl -n '__kubectl_needs_cmd' -xA \
         -a 'get describe create replace patch delete logs rolling-update scale 
             exec port-forward proxy run stop expose label config cluster-info \
             api-versions version help' 


# ===================================================================================
# kubectl get *
# ===================================================================================

complete -c kubectl -n '__kubectl_using_cmd get'      -l all-namespaces
complete -c kubectl -n '__kubectl_using_cmd get' -s h -l help
complete -c kubectl -n '__kubectl_using_cmd get' -s L -l label-columens
complete -c kubectl -n '__kubectl_using_cmd get'      -l no-headers
complete -c kubectl -n '__kubectl_using_cmd get' -s o -l output -xA -a 'json yaml template templatefile wide' 
complete -c kubectl -n '__kubectl_using_cmd get'      -l output-version 
complete -c kubectl -n '__kubectl_using_cmd get' -s l -l selector
complete -c kubectl -n '__kubectl_using_cmd get' -s t -l template
complete -c kubectl -n '__kubectl_using_cmd get' -s w -l watch
complete -c kubectl -n '__kubectl_using_cmd get'      -l watch-only


# ===================================================================================
# kubectl describe *
# ===================================================================================

complete -c kubectl -n '__kubectl_using_cmd describe' -s h -l help
complete -c kubectl -n '__kubectl_using_cmd describe' -s l -l selector


# ===================================================================================
# kubectl create *
# ===================================================================================

complete -c kubectl -n '__kubectl_using_cmd create' -s f -l filename
complete -c kubectl -n '__kubectl_using_cmd create' -s h -l help


# ===================================================================================
# kubectl replace *
# ===================================================================================

complete -c kubectl -n '__kubectl_using_cmd replace'      -l cascade
complete -c kubectl -n '__kubectl_using_cmd replace' -s f -l filename
complete -c kubectl -n '__kubectl_using_cmd replace'      -l force
complete -c kubectl -n '__kubectl_using_cmd replace'      -l grace-period
complete -c kubectl -n '__kubectl_using_cmd replace' -s h -l help
complete -c kubectl -n '__kubectl_using_cmd replace'      -l timeout


# ===================================================================================
# kubectl patch *
# ===================================================================================

complete -c kubectl -n '__kubectl_using_cmd patch' -s h -l help
complete -c kubectl -n '__kubectl_using_cmd patch' -s p -l patch


# ===================================================================================
# kubectl delete *
# ===================================================================================

complete -c kubectl -n '__kubectl_using_cmd delete'      -l all
complete -c kubectl -n '__kubectl_using_cmd delete'      -l cascade
complete -c kubectl -n '__kubectl_using_cmd delete' -s f -l filename
complete -c kubectl -n '__kubectl_using_cmd delete'      -l grace-period
complete -c kubectl -n '__kubectl_using_cmd delete' -s h -l help
complete -c kubectl -n '__kubectl_using_cmd delete'      -l ignore-not-found
complete -c kubectl -n '__kubectl_using_cmd delete' -s l -l selector
complete -c kubectl -n '__kubectl_using_cmd delete'      -l timeout


# ===================================================================================
# kubectl logs *
# ===================================================================================

complete -c kubectl -n "__kubectl_needs_subcmd logs" -xA -a "(__kubectl_get pods)" 
complete -c kubectl -n "__kubectl_needs_subsubcmd logs __whatever" -xA -a "(__kubectl_containers)" 

complete -c kubectl -n '__kubectl_using_cmd logs' -s c -l container -xA -a "(__kubectl_containers)" 
complete -c kubectl -n '__kubectl_using_cmd logs' -s f -l follow
complete -c kubectl -n '__kubectl_using_cmd logs' -s h -l help
complete -c kubectl -n '__kubectl_using_cmd logs'      -l interactive
complete -c kubectl -n '__kubectl_using_cmd logs' -s p -l previous


# ===================================================================================
# kubectl rolling-update *
# ===================================================================================

complete -c kubectl -n "__kubectl_needs_subcmd rolling-update" -xA -a "(__kubectl_get rc)" 

complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l deployment-label-key
complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l dry-run
complete -c kubectl -n '__kubectl_using_cmd rolling-update' -s f -l filename
complete -c kubectl -n '__kubectl_using_cmd rolling-update' -s h -l help 
complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l image
complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l no-headers
complete -c kubectl -n '__kubectl_using_cmd rolling-update' -s o -l output -xA -a 'json yaml template templatefile wide' 
complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l output-version
complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l poll-interval
complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l rollback
complete -c kubectl -n '__kubectl_using_cmd rolling-update' -s t -l template
complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l timeout
complete -c kubectl -n '__kubectl_using_cmd rolling-update'      -l upgrade-period


# ===================================================================================
# kubectl scale *
# ===================================================================================

complete -c kubectl -n "__kubectl_needs_subcmd scale" -xA -a "replicationcontrollers"
complete -c kubectl -n "__kubectl_needs_subsubcmd scale replicationcontrollers" -xA -a "(__kubectl_get replicationcontrollers)" 
complete -c kubectl -n '__kubectl_using_cmd sacle'      -l current-replicas
complete -c kubectl -n '__kubectl_using_cmd scale' -s h -l help 
complete -c kubectl -n '__kubectl_using_cmd sacle'      -l replicas
complete -c kubectl -n '__kubectl_using_cmd sacle'      -l resource-version


# ===================================================================================
# kubectl exec *
# ===================================================================================

complete -c kubectl -n "__kubectl_needs_subcmd exec" -xA -a "(__kubectl_get pods)" 
complete -c kubectl -n "__kubectl_needs_subsubcmd exec __whatever" -xA -a "(__kubectl_containers)" 

complete -c kubectl -n '__kubectl_using_cmd exec' -s c -l container -xA -a "(__kubectl_containers)" 
complete -c kubectl -n '__kubectl_using_cmd exec' -s h -l help 
complete -c kubectl -n '__kubectl_using_cmd exec' -s i -l stdin
complete -c kubectl -n '__kubectl_using_cmd exec' -s t -l tty


# ===================================================================================
# kubectl expose *
# ===================================================================================

complete -c kubectl -n "__kubectl_needs_subcmd expose" -xA -a 'service rc'
complete -c kubectl -n "__kubectl_needs_subsubcmd expose rc" -xA -a "(__kubectl_get rc)" 
complete -c kubectl -n "__kubectl_needs_subsubcmd expose service" -xA -a "(__kubectl_get service)" 

complete -c kubectl -n '__kubectl_using_cmd expose'      -l container-port
complete -c kubectl -n '__kubectl_using_cmd expose'      -l create-external-load-balancer
complete -c kubectl -n '__kubectl_using_cmd expose'      -l dry-run
complete -c kubectl -n '__kubectl_using_cmd expose'      -l generator -xA -a "service/v1 service/v2"
complete -c kubectl -n '__kubectl_using_cmd expose' -s h -l help 
complete -c kubectl -n '__kubectl_using_cmd expose' -s l -l labels 
complete -c kubectl -n '__kubectl_using_cmd expose'      -l name 
complete -c kubectl -n '__kubectl_using_cmd expose'      -l no-headers
complete -c kubectl -n '__kubectl_using_cmd expose' -s o -l output -xA -a 'json yaml template templatefile wide' 
complete -c kubectl -n '__kubectl_using_cmd expose'      -l output-version
complete -c kubectl -n '__kubectl_using_cmd expose'      -l overrides
complete -c kubectl -n '__kubectl_using_cmd expose'      -l port
complete -c kubectl -n '__kubectl_using_cmd expose'      -l protocol
complete -c kubectl -n '__kubectl_using_cmd expose'      -l public-ip
complete -c kubectl -n '__kubectl_using_cmd expose'      -l selector
complete -c kubectl -n '__kubectl_using_cmd expose'      -l target-port
complete -c kubectl -n '__kubectl_using_cmd expose'      -l template
complete -c kubectl -n '__kubectl_using_cmd expose'      -l type -xA -a 'ClusterIP NodePort LoadBalancer'


# ===================================================================================
# kubectl port-forward *
# ===================================================================================

complete -c kubectl -n "__kubectl_using_cmd port-forward" -s p -l pod -xA -a "(__kubectl_get pods)" 
complete -c kubectl -n '__kubectl_using_cmd port-forward' -s h -l help 


# ===================================================================================
# kubectl proxy *
# ===================================================================================

complete -c kubectl -n "__kubectl_using_cmd proxy"      -l accept-hosts
complete -c kubectl -n "__kubectl_using_cmd proxy"      -l accept-paths
complete -c kubectl -n "__kubectl_using_cmd proxy"      -l api-prefix
complete -c kubectl -n "__kubectl_using_cmd proxy"      -l disable-filter
complete -c kubectl -n '__kubectl_using_cmd proxy' -s h -l help 
complete -c kubectl -n "__kubectl_using_cmd proxy" -s p -l port
complete -c kubectl -n "__kubectl_using_cmd proxy"      -l reject-methods
complete -c kubectl -n "__kubectl_using_cmd proxy"      -l reject-paths
complete -c kubectl -n "__kubectl_using_cmd proxy" -s w -l www
complete -c kubectl -n "__kubectl_using_cmd proxy" -s P -l www-prefix


# ===================================================================================
# kubectl run *
# ===================================================================================

complete -c kubectl -n "__kubectl_using_cmd run"      -l dry-run
complete -c kubectl -n "__kubectl_using_cmd run"      -l generator
complete -c kubectl -n '__kubectl_using_cmd run' -s h -l help 
complete -c kubectl -n "__kubectl_using_cmd run"      -l hostport
complete -c kubectl -n "__kubectl_using_cmd run"      -l image
complete -c kubectl -n "__kubectl_using_cmd run" -s l -l labels
complete -c kubectl -n "__kubectl_using_cmd run"      -l no-headers 
complete -c kubectl -n '__kubectl_using_cmd run' -s o -l output -xA -a 'json yaml template templatefile wide' 
complete -c kubectl -n '__kubectl_using_cmd run'      -l output-version
complete -c kubectl -n '__kubectl_using_cmd run'      -l overrides
complete -c kubectl -n '__kubectl_using_cmd run'      -l port
complete -c kubectl -n "__kubectl_using_cmd run" -s r -l replicas
complete -c kubectl -n "__kubectl_using_cmd run" -s t -l template


# ===================================================================================
# kubectl stop *
# ===================================================================================

complete -c kubectl -n "__kubectl_using_cmd stop"      -l all
complete -c kubectl -n "__kubectl_using_cmd stop" -s f -l filename 
complete -c kubectl -n "__kubectl_using_cmd stop"      -l grace-period
complete -c kubectl -n "__kubectl_using_cmd stop" -s h -l help
complete -c kubectl -n "__kubectl_using_cmd stop"      -l ignore-not-found
complete -c kubectl -n "__kubectl_using_cmd stop" -s l -l selector
complete -c kubectl -n "__kubectl_using_cmd stop"      -l timeout


# ===================================================================================
# kubectl label *
# ===================================================================================

complete -c kubectl -n "__kubectl_using_cmd label"      -l all
complete -c kubectl -n "__kubectl_using_cmd label" -s h -l help
complete -c kubectl -n "__kubectl_using_cmd label"      -l no-headers
complete -c kubectl -n '__kubectl_using_cmd label' -s o -l output -xA -a 'json yaml template templatefile wide' 
complete -c kubectl -n '__kubectl_using_cmd label'      -l output-version
complete -c kubectl -n '__kubectl_using_cmd label'      -l overwrite
complete -c kubectl -n '__kubectl_using_cmd label'      -l resource-version
complete -c kubectl -n "__kubectl_using_cmd label" -s l -l selector
complete -c kubectl -n "__kubectl_using_cmd label" -s t -l template


# ===================================================================================
# kubectl config *
# ===================================================================================

complete -c kubectl -n "__kubectl_needs_subcmd config" -xA -a 'view set-cluster set-credentials set-context set unset use-context'

complete -c kubectl -n "__kubectl_using_cmd config" -s h -l help
complete -c kubectl -n "__kubectl_using_cmd config"      -l kubeconfig

complete -c kubectl -n "__kubectl_using_subcmd config view"      -l flatten
complete -c kubectl -n "__kubectl_using_subcmd config view" -s h -l help
complete -c kubectl -n "__kubectl_using_subcmd config view"      -l merge
complete -c kubectl -n "__kubectl_using_subcmd config view"      -l minify
complete -c kubectl -n "__kubectl_using_subcmd config view"      -l no-headers
complete -c kubectl -n '__kubectl_using_subcmd config view' -s o -l output -xA -a 'json yaml template templatefile wide' 
complete -c kubectl -n '__kubectl_using_subcmd config view'      -l output-version
complete -c kubectl -n '__kubectl_using_subcmd config view'      -l raw
complete -c kubectl -n '__kubectl_using_subcmd config view'      -l template

complete -c kubectl -n "__kubectl_needs_subsubcmd config set-cluster" -xA -a "(__kubectl_clusters)" 
complete -c kubectl -n "__kubectl_using_subcmd config set-cluster"      -l api-version
complete -c kubectl -n "__kubectl_using_subcmd config set-cluster" -s h -l help
complete -c kubectl -n "__kubectl_using_subcmd config set-cluster"      -l certificate-authority
complete -c kubectl -n "__kubectl_using_subcmd config set-cluster"      -l embed-certs
complete -c kubectl -n "__kubectl_using_subcmd config set-cluster" -s h -l help
complete -c kubectl -n "__kubectl_using_subcmd config set-cluster"      -l insecure-skip-tls-verify
complete -c kubectl -n "__kubectl_using_subcmd config set-cluster"      -l server

complete -c kubectl -n "__kubectl_needs_subsubcmd config set-credentials" -xA -a "(__kubectl_users)" 
complete -c kubectl -n "__kubectl_using_subcmd config set-credentials"      -l client-certificate
complete -c kubectl -n "__kubectl_using_subcmd config set-credentials"      -l client-key
complete -c kubectl -n "__kubectl_using_subcmd config set-credentials"      -l embed-certs
complete -c kubectl -n "__kubectl_using_subcmd config set-credentials" -s h -l help
complete -c kubectl -n "__kubectl_using_subcmd config set-credentials"      -l password
complete -c kubectl -n "__kubectl_using_subcmd config set-credentials"      -l token
complete -c kubectl -n "__kubectl_using_subcmd config set-credentials"      -l username

complete -c kubectl -n "__kubectl_needs_subsubcmd config set-context" -xA -a "(__kubectl_contexts)" 
complete -c kubectl -n "__kubectl_using_subcmd config set-context"      -l cluster -xA -a "(__kubectl_clusters)"
complete -c kubectl -n "__kubectl_using_subcmd config set-context" -s h -l help
complete -c kubectl -n "__kubectl_using_subcmd config set-context"      -l namespace -xA -a "(__kubectl_get namespaces)"
complete -c kubectl -n "__kubectl_using_subcmd config set-context"      -l user -xA -a "(__kubectl_users)"

complete -c kubectl -n "__kubectl_using_subcmd config set" -s h -l help
complete -c kubectl -n "__kubectl_using_subcmd config unset" -s h -l help

complete -c kubectl -n "__kubectl_needs_subsubcmd config use-context" -xA -a "(__kubectl_contexts)" 
complete -c kubectl -n "__kubectl_using_subcmd config use-context" -s h -l help


# ===================================================================================
# kubectl cluster-info *
# ===================================================================================

complete -c kubectl -n "__kubectl_using_cmd cluster-info" -s h -l help


# ===================================================================================
# kubectl api-versions *
# ===================================================================================

complete -c kubectl -n "__kubectl_using_cmd api-versions" -s h -l help


# ===================================================================================
# kubectl version *
# ===================================================================================

complete -c kubectl -n "__kubectl_using_cmd version" -s c -l client
complete -c kubectl -n "__kubectl_using_cmd version" -s h -l help
