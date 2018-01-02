set +x
set +e 

echo ""
echo ""
echo "---"
echo "Now starting POST-BUILD steps"
echo "---"
echo ""

echo INFO: Pointing to $DOCKER_HOST

if [ ! $(docker ps -aq | wc -l) -eq 0 ]; then
	echo INFO: Removing containers...
	! docker rm -vf $(docker ps -aq)
fi

# Remove all images which don't have docker or debian in the name
if [ ! $(docker images | sed -n '1!p' | grep -v 'docker' | grep -v 'debian' | awk '{ print $3 }' | wc -l) -eq 0 ]; then 
	echo INFO: Removing images...
	! docker rmi -f $(docker images | sed -n '1!p' | grep -v 'docker' | grep -v 'debian' | awk '{ print $3 }') 
fi

# Kill off any instances of git, go and docker, just in case
! taskkill -F -IM git.exe -T >& /dev/null
! taskkill -F -IM go.exe -T >& /dev/null
! taskkill -F -IM docker.exe -T >& /dev/null

# Remove everything
! cd /c/jenkins/gopath/src/github.com/docker/docker
! rm -rfd * >& /dev/null
! rm -rfd .* >& /dev/null

echo INFO: Cleanup complete
exit 0