#!/bin/bash -e

leader=http://10.240.201.15:2379
# assume three servers
servers=( http://10.240.201.15:2379 http://10.240.212.209:2379 http://10.240.95.3:2379 )

keyarray=( 64 256 )

for keysize in ${keyarray[@]}; do

  echo write, 1 client, $keysize key size, to leader
  ./hey -m PUT -n 10 -d value=`head -c $keysize < /dev/zero | tr '\0' '\141'` -c 1 -T application/x-www-form-urlencoded $leader/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo

  echo write, 64 client, $keysize key size, to leader
  ./hey -m PUT -n 640 -d value=`head -c $keysize < /dev/zero | tr '\0' '\141'` -c 64 -T application/x-www-form-urlencoded $leader/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo

  echo write, 256 client, $keysize key size, to leader
  ./hey -m PUT -n 2560 -d value=`head -c $keysize < /dev/zero | tr '\0' '\141'` -c 256 -T application/x-www-form-urlencoded $leader/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo

  echo write, 64 client, $keysize key size, to all servers
  for i in ${servers[@]}; do
    ./hey -m PUT -n 210 -d value=`head -c $keysize < /dev/zero | tr '\0' '\141'` -c 21 -T application/x-www-form-urlencoded $i/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo &
  done
  # wait for all heys to start running
  sleep 3
  # wait for all heys to finish
  for pid in $(pgrep 'hey'); do
    while kill -0 "$pid" 2> /dev/null; do
      sleep 3
    done
  done

  echo write, 256 client, $keysize key size, to all servers
  for i in ${servers[@]}; do
    ./hey -m PUT -n 850 -d value=`head -c $keysize < /dev/zero | tr '\0' '\141'` -c 85 -T application/x-www-form-urlencoded $i/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo &
  done
  sleep 3
  for pid in $(pgrep 'hey'); do
    while kill -0 "$pid" 2> /dev/null; do
      sleep 3
    done
  done

  echo read, 1 client, $keysize key size, to leader
  ./hey -n 100 -c 1 $leader/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo

  echo read, 64 client, $keysize key size, to leader
  ./hey -n 6400 -c 64 $leader/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo

  echo read, 256 client, $keysize key size, to leader
  ./hey -n 25600 -c 256 $leader/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo

  echo read, 64 client, $keysize key size, to all servers
  # bench servers one by one, so it doesn't overload this benchmark machine
  # It doesn't impact correctness because read request doesn't involve peer interaction.
  for i in ${servers[@]}; do
    ./hey -n 21000 -c 21 $i/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo
  done

  echo read, 256 client, $keysize key size, to all servers
  for i in ${servers[@]}; do
    ./hey -n 85000 -c 85 $i/v2/keys/foo | grep -e "Requests/sec" -e "Latency" -e "90%" | tr "\n" "\t" | xargs echo
  done

done
