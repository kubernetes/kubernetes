FROM ubuntu:14.04

# Install all the required packages.
RUN apt-get update
RUN apt-get -y install \
  git apache2 dpkg-dev python-pygments \
  php5 php5-mysql php5-gd php5-dev php5-curl php-apc php5-cli php5-json php5-xhprof
RUN a2enmod rewrite
RUN apt-get source php5
RUN (cd `ls -1F | grep '^php5-.*/$'`/ext/pcntl && phpize && ./configure && make && sudo make install)

# Load code source.
RUN mkdir /home/www-data
RUN cd /home/www-data && git clone https://github.com/phacility/libphutil.git
RUN cd /home/www-data && git clone https://github.com/phacility/arcanist.git
RUN cd /home/www-data && git clone https://github.com/phacility/phabricator.git
RUN chown -R www-data /home/www-data
RUN chgrp -R www-data /home/www-data

ADD 000-default.conf /etc/apache2/sites-available/000-default.conf
ADD run.sh /run.sh
RUN chmod a+x /*.sh

# Run Apache2.
EXPOSE 80
CMD ["/run.sh"]
