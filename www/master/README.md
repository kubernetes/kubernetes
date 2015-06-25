### Application source and project files

This directory contains the source and project files for the application, including:

* `bower.json`, which declares the framework dependencies downloaded by bower, 
* `gulpfile.js`, which defines the build tasks run by `npm start` and `npm run build`, 
* `karma.conf.js`, the default configuration file for top level karma tests.
* `package.json`, which declares the tool dependences downloaded by `npm install`.
* `vendor.json` and `vendor.base.json`, which declare dependencies compiled into `app.js` and `base.js`, respectively.

You will find the following directories inside:

* `components/` This directory contains components that appear as tabs in the application. See [master/components/README.md](master/components/README.md).
* `less/` This directory contains the LESS files for the core styles and material styles.
* `js/` Here you will find JavaScript files compiled into `app.js`.
* `protractor/` This directory contains the default  configuration file `conf.js` and the `*.spec.js` files for  the top level protractor tests.
* `shared/` This directory contains assets shared by two or more components.
* `test/` This directory contains the `*.spec.js` files for  the top level karma tests. The default configuration file is `master/karma.conf.js`.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/www/master/README.md?pixel)]()
