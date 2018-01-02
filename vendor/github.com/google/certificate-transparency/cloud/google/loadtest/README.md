Load Testing
============

This directory contains some scripts and configs to let you easily perform loadtests against your CT log/mirror instance.

The load itself is generated with [curl-loader](http://curl-loader.sourceforge.net/index.html) running on one or more GCE instances.

HOWTO run a load test
=====================

1. Create a new GCE project in which to run your load generator instances.
1. Download curl-loader from [here](http://sourceforge.net/project/showfiles.php?group_id=179599)
1. Build curl-loader.  By default the GCE instance is running an ubuntu-14-10 image, so you'll want to build the curl-loader binary on something similar.
1. Copy your freshly built curl-loader into GCS:

   ```bash
   gcloud config set project YOUR_PROJECT_NAME_HERE
   gsutil cp path/to/curl-loader gs://YOUR_PROJECT_NAME_HERE/
   ```

1. Create a config file, there are examples in `cloud/google/loadtest/configs`, in particular make sure that you set the `PROJECT` variable to the name of your newly cretaed project, and set `TARGET` to the public IP address/hostname of the log (or mirror) you want to load test.
1. You can customise the path which is requested, and even create 'session scripts' if you wish, see the [curl-loader documentation](http://curl-loader.sourceforge.net/doc/faq.html#configuration) for more details, and update the `LOADER_CONF` variable with your new curl-loader config script.
1. Run `start_load.sh` to start the load testing, this will spin up new GCE instances so there'll be a small delay before the load starts.
1. Do other stuff.
1. If you are interested, the curl-loader stats are output to `/var/log/startup-script` log files on the load generating GCE instances.
1. Once you've finished beating on the log instance, turn the load generators off by running the `stop_load.sh` script.


