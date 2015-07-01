FROM ubuntu:trusty

# update the package repository and install python pip
RUN apt-get -y update && apt-get -y install python-dev python-pip

# install flower
RUN pip install flower

# Make sure we expose port 5555 so that we can connect to it
EXPOSE 5555

ADD run_flower.sh /usr/local/bin/run_flower.sh

# Running flower
CMD ["/bin/bash", "/usr/local/bin/run_flower.sh"]
