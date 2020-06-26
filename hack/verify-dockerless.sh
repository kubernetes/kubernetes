not_have_dockerless=$(find ../pkg/kubelet/dockershim -name "*.go" ! -exec grep -q 'dockerless' {} \; -print)
if [[ $not_have_dockerless ]]; then
        echo "Following files do not contain dockerless tag"
        echo $not_have_dockerless
fi
imports_docker=$(find ../{cmd,pkg}/kubelet/ -name "*.go" -exec grep -q 'github.com/docker/docker' {} \; -print | grep -v ../pkg/kubelet/dockershim)
if [[ $imports_docker ]]; then
        echo "Following files imports docker outside dockershim"
        echo $imports_docker
fi
imports_dockershim=$(find ../{cmd,pkg}/kubelet/ -name "*.go" -exec grep -q 'kubelet/dockershim' {} \; -print | grep -v dockershim)
if [[ $imports_dockershim ]]; then
        echo "Following files imports dockershim outside dockershim folder and kubelet_dockershim.go"
        echo $imports_dockershim
fi
