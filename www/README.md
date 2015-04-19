### Install dependencies

We have two kinds of dependencies in this project: tools and angular framework code.  The tools help
us manage and test the application.

* We get the tools we depend upon via `npm`, the [node package manager](https://www.npmjs.com/).
* We get the angular code via `bower`, a [client-side code package manager](http://bower.io/).

`npm` is configured to automatically run `bower install` and `gulp`. Before you run the application for the first time, simply run this command from the `www/master` directory:

```
npm install
```

To start the application, run this command from the `www/master` directory:

```
npm start
```

The `gulp` command will start a file watcher which will update the generated `app` code after any changes are saved. Note: gulp file watcher does not currently support adding or deleting files, this will require a restart of gulp). Two new directories will also be created in the project.

* `master/node_modules` - contains npm dependencies
* `master/bower_components` - contains the angular framework files and any custom dependencies

Bower components should be refernced in one of the `vendor.json` files below:

* `master/vendor.base.json` - 3rd party vendor javascript required to start the app. JS is compiled to `base.js` and loaded before `app.js`
* `master/vendor.json` - 3rd party vendor scripts to make the app work, usually lazy loaded. Can be js or css. Copied to `vendor/*`.

### Serving the app during development

The app can be served through `kubectl`, but for some types of review a local web server is convenient. One can be installed as follows:

```
sudo npm install -g http-server
```

The server can then be launched:

```
cd app
http-server -a localhost -p 8000
```

### Configuration
#### Configuration settings
A json file can be used by `gulp` to automatically create angular constants. This is useful for setting per environment variables such as api endpoints.
*  ```www/master/shared/config/development.json``` or ```www/master/shared/config/production.json``` can be created from the ```www/master/shared/config/development.example.json``` file.
* ```development.example.json``` should be kept up to date with default values, since ```development.json``` is not under source control.
* Component configuration can be added to ```www/master/components/<component name>/config/development.json``` and it will be combined with the main app config files and compiled into the intermediary ```www/master/shared/config/generated-config.js``` file.
* All ```generated-config.js``` is compiled into ```app.js```
* Production config can be generated using ```gulp config --env production``` or ```gulp --env production```
* The generated angular constant is named ```ENV``` with the shared root and each component having their own child configuration. For example,
```
www/master
├── shared/config/development.json
└── components
    ├── dashboard/config/development.json
    ├── graph/config/development.json
    └── my_component/config/development.json
```
produces ```www/master/shared/config/generated-config.js```:
```
angular.module('kubernetesApp.config', [])
.constant('ENV', {
  '/': <www/master/shared/config/development.json>,
  'dashboard': <www/master/components/dashboard/config/development.json>,
  'graph': <www/master/components/graph/config/development.json>,
  'my_component': <www/master/components/my_component/config/development.json>
});
```

#### Kubernetes server configuration

**RECOMMENDED**: By default the Kubernetes api server does not support CORS,
  so the `kube-apiserver.service` must be started with
  `--cors_allowed_origins=.*` or `--cors_allowed_origins=http://<your
  host here>`

**HACKS**: If you don't want to/cannot restart the Kubernetes api server:
* Or you can start your browser with web security disabled. For
  Chrome, you can [launch](http://www.chromium.org/developers/how-tos/run-chromium-with-flags) it with flag ```--disable-web-security```.

### Building a new visualizer or component

See [master/components/README.md](master/components/README.md).

### Testing
Currently kuberntes-ui includes both unit-testing (run via [Karma](http://karma-runner.github.io/0.12/index.html)) and
end-to-end testing (run via
[Protractor](http://angular.github.io/protractor/#/)).

#### Unittests via Karma
To run the existing Karma tests:
* Install the Karma CLI: `sudo npm install -g karma-cli` (it needs to
  be installed globally, hence the `sudo` may be needed). Note that
  the other Karma packages (such as `karma`, `karma-jasmine`, and
  `karma-chrome-launcher` should be automatically installed when
  running `npm start`).
* Go to the `www/master` directory, and run `karma start
karma.conf.js`. The Karma configuration is defined in `karma.config.js`. The console should show the test results.

To write new Karma tests:
* For testing each components, write test files (`*.spec.js`) under the
corresponding `www/master/components/**/test/modules/` directory.
* For testing the chrome and the framework, write test files
  (*.spec.js) under the `www/master/test/modules/*` directory.

#### End-to-end testing via Protractor
To run the existing Protractor tests:
* Install the CLIs: `sudo npm install -g protractor`.
* Start the webdriver server: `sudo webdriver-manager start`
* Start the kubernetes-ui app (see instructions above), assuming
running at port 8000.
* Go to the `www/master/protractor` directory and run `protractor
  conf.js`. The protractor configuration is in `conf.js`. The console
  should show the test results.

To write new protractor tests, put the test files (`*.spec.js`) in the
corresponding `www/master/components/**/protractor/` directory.
