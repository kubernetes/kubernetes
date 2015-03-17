## First set up the host VM.  That ensures
## we avoid vagrant race conditions.
set -x 

cd hosts/ 
echo "note: the VM must be running before you try this"
echo "if not already running, cd to hosts and run vagrant up"
vagrant provision
#echo "removing containers"
#vagrant ssh -c "sudo docker rm -f $(docker ps -a -q)"
cd ..

## Now spin up the docker containers
## these will run in the ^ host vm above.

vagrant up

## Finally, curl the length, it should be 3 .

x=`curl localhost:3000/llen`

for i in `seq 1 100` do
    if [ x$x == "x3" ]; then 
       echo " passed $3 "
       exit 0
    else
       echo " FAIL" 
    fi
done

exit 1 # if we get here the test obviously failed.
