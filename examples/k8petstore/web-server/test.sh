echo "start test of frontend"
curl localhost:3000/llen
curl localhost:3000/llen
curl localhost:3000/llen
curl localhost:3000/llen
curl localhost:3000/llen
curl localhost:3000/llen
x=`curl localhost:3000/llen`
echo "done testing frontend result = $x"
