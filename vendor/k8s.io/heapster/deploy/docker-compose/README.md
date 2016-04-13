# Quick start with docker-compose

Make sure you have installed [Docker Compose](https://docs.docker.com/compose/).  Once compose is installed go ahead and fire the awesome:

    $ docker-compose up

InfluxDB web UI is on http://localhost:8086 and Grafana is available on http://localhost:3000. *Note*: if you are using [boot2docker](https://github.com/boot2docker/boot2docker) you need to replace localhost by the one provided by boot2docker, example:
    
    $ boot2docker ip
    192.168.59.103
    $

in this case you will need to visit http://192.168.59.103:8086 (InfluxDB) and http://192.168.59.103:3000 (Grafana)

The provided compose it's configured by default with the cAdvisor source and will setup to poll metrics from the `cAdvisor` host defined in sample-hosts.json.
