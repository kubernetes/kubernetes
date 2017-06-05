# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
