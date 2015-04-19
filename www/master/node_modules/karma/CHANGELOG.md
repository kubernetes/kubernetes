<a name="v0.10.9"></a>
### v0.10.9 (2013-12-29)


#### Features

* **deps:** update chokidar ([26792e44](http://github.com/karma-runner/karma/commit/26792e44c32e39c14e8e90a01573438adbd85dae), closes [#867](http://github.com/karma-runner/karma/issues/867))

<a name="v0.10.8"></a>
### v0.10.8 (2013-12-04)


#### Bug Fixes

* **preprocess:** set correct extension for the preprocessed path ([69cd4ebc](http://github.com/karma-runner/karma/commit/69cd4ebc4c0d9b1ec31b71b0c1c554ea52e1aae7), closes [#843](http://github.com/karma-runner/karma/issues/843))

<a name="v0.10.7"></a>
### v0.10.7 (2013-12-01)


#### Bug Fixes

* **config:**
  * ignore empty string patterns ([cd045943](http://github.com/karma-runner/karma/commit/cd0459432fb66c66245c5f00ddde3a576ddca1d2))
  * apply CLI logger options as soon as we can ([6a02056c](http://github.com/karma-runner/karma/commit/6a02056ca45752211abfb4816d39f5f583fda9a4))

<a name="v0.10.6"></a>
### v0.10.6 (2013-11-26)


#### Bug Fixes

* **browser:** reply "start" event ([8c1feae1](http://github.com/karma-runner/karma/commit/8c1feae1c0c60077365ca977fcf96640da449bee))


#### Features

* **launcher:** send SIGKILL if SIGINT does not kill the browser ([eefcf00d](http://github.com/karma-runner/karma/commit/eefcf00dd869c7fce4da86064c2f063b5db39372))

<a name="v0.10.5"></a>
### v0.10.5 (2013-11-20)


#### Bug Fixes

* **config:** not append empty module if no custom launcher/rep/prep ([c025cdcd](http://github.com/karma-runner/karma/commit/c025cdcdd97b8b9efc0c0dc1086b6c3a31aadebc))
* **watcher:** allow parentheses in a pattern ([8eaa0562](http://github.com/karma-runner/karma/commit/8eaa0562e07e8ab06fea04f134bdfbdffc2e84f6), closes [#728](http://github.com/karma-runner/karma/issues/728))


#### Features

* **config:** log if no config file is specified ([27a07912](http://github.com/karma-runner/karma/commit/27a079129f5adcc0478b6006ee69d3323faa4431))

<a name="v0.10.4"></a>
### v0.10.4 (2013-10-25)

<a name="v0.10.3"></a>
### v0.10.3 (2013-10-25)


#### Bug Fixes

* **static:** Use full height for the iFrame. Fix based on PR #714. ([ca98f3a4](http://github.com/karma-runner/karma/commit/ca98f3a4eafaf9aa3e84636364d108ffa49bc9e3))
* **watcher:**
  * ignore double "add" events ([8a55901b](http://github.com/karma-runner/karma/commit/8a55901be322586b3a747be7c36b2dd6f6dd9923))
  * improve watching efficiency ([1e4a66d3](http://github.com/karma-runner/karma/commit/1e4a66d33aeaec7900baeee0867679b9cbc3e535), closes [#616](http://github.com/karma-runner/karma/issues/616))


#### Features

* **watcher:** ignore initial "add" events ([7ab9e7bd](http://github.com/karma-runner/karma/commit/7ab9e7bd442ac4fdad6eadb37dbb69f1837c6434))

<a name="v0.10.2"></a>
### v0.10.2 (2013-08-21)


#### Bug Fixes

* don't mark a browser captured if already being killed/timeouted ([21230979](http://github.com/karma-runner/karma/commit/212309795861cf599dbcc0ed60fff612ccf25cf5), closes [#88](http://github.com/karma-runner/karma/issues/88))


#### Features

* sync page unload (disconnect) ([ac9b3f01](http://github.com/karma-runner/karma/commit/ac9b3f01e88ce2cf91fc86aca9cecfdb8177a6fa))
* buffer result messages when polling ([c4ad6970](http://github.com/karma-runner/karma/commit/c4ad69709103110a066ae1d9652af69e42434c6b))
* allow browser to reconnect during the test run ([cbe2851b](http://github.com/karma-runner/karma/commit/cbe2851baa55312f00be420e0345283b33326266), closes [#82](http://github.com/karma-runner/karma/issues/82), [#590](http://github.com/karma-runner/karma/issues/590))

<a name="v0.10.1"></a>
### v0.10.1 (2013-08-06)


#### Bug Fixes

* **cli:** Always pass an instance of fs to processArgs. ([06532b70](http://github.com/karma-runner/karma/commit/06532b7042371f270c227a1a7f859f2dab5afac1), closes [#677](http://github.com/karma-runner/karma/issues/677))
* **init:** set default filename ([34d49b13](http://github.com/karma-runner/karma/commit/34d49b138f3bee8f17e1e9e343012d82887f906b), closes [#680](http://github.com/karma-runner/karma/issues/680), [#681](http://github.com/karma-runner/karma/issues/681))

<a name="v0.10.0"></a>
## v0.10.0 (2013-08-06)

<a name="v0.9.8"></a>
### v0.9.8 (2013-08-05)


#### Bug Fixes

* **init:** install plugin as dev dependency ([46b7a402](http://github.com/karma-runner/karma/commit/46b7a402fb8d700b10e2d72908c309d27212b5a0))
* **runner:** do not confuse client args with the config file ([6f158aba](http://github.com/karma-runner/karma/commit/6f158abaf923dad6878a64da2d8a3c2c56ae604f))


#### Features

* **config:** default config can be karma.conf.js or karma.conf.coffee ([d4a06f29](http://github.com/karma-runner/karma/commit/d4a06f296c4d805f2dccd85b4898766593af4d66))
* **runner:**
  * support config files ([449e4a1a](http://github.com/karma-runner/karma/commit/449e4a1ad8b8543f84f1953c875cfbdf5692caa7), closes [#625](http://github.com/karma-runner/karma/issues/625))
  * add --no-refresh to disable re-globbing ([b9c670ac](http://github.com/karma-runner/karma/commit/b9c670accbde8d027bdc3e09a4080c546b05853c))

<a name="v0.9.7"></a>
### v0.9.7 (2013-07-31)


#### Bug Fixes

* **init:** trim the inputs ([b72355cb](http://github.com/karma-runner/karma/commit/b72355cbeadc8e907e48bbd7d9a11e6de17343f7), closes [#663](http://github.com/karma-runner/karma/issues/663))
* **web-server:** correct urlRegex in custom handlers ([a641c2c1](http://github.com/karma-runner/karma/commit/a641c2c1dd0f5f1e0045e7cff1516d2820a8204e))


#### Features

* basic bash/zsh completion ([9dc1cf6a](http://github.com/karma-runner/karma/commit/9dc1cf6a6e095653fed6c79c4896c71af8af1953))
* **runner:** allow passing changed/added/removed files ([b598106d](http://github.com/karma-runner/karma/commit/b598106de1295f3e1e58338a8eca2b60f99175c3))
* **watcher:** make the batching delay configurable ([fa139312](http://github.com/karma-runner/karma/commit/fa139312a0fff981f11182c17ba6979dccca1105))

<a name="v0.9.6"></a>
### v0.9.6 (2013-07-28)


#### Features

* pass command line opts through to browser ([00d63d0b](http://github.com/karma-runner/karma/commit/00d63d0b965a998b04d1917d4c4421abc24cec18))
* **web-server:** compress responses (gzip/deflate) ([8e8a2d44](http://github.com/karma-runner/karma/commit/8e8a2d4418e7abef7dca42e58bf09c95b07687b2))


#### Breaking Changes

* `runnerPort` is merged with `port`
if you are using `karma run` with custom `--runer-port`, please change that to `--port`.
 ([ca4c4d88](http://github.com/karma-runner/karma/commit/ca4c4d88b9a4a1992f7975aa32b37a008394847b))

<a name="v0.9.5"></a>
### v0.9.5 (2013-07-21)


#### Bug Fixes

* detect a full page reload, show error and recover ([15d80f47](http://github.com/karma-runner/karma/commit/15d80f47a227839e9b0d54aeddf49b9aa9afe8aa), closes [#27](http://github.com/karma-runner/karma/issues/27))
* better serialization in dump/console.log ([fd46365d](http://github.com/karma-runner/karma/commit/fd46365d1fd3a9bea15c04abeb7df33a3a2d96a4), closes [#640](http://github.com/karma-runner/karma/issues/640))
* browsers_change event always has collection as arg ([42bf787f](http://github.com/karma-runner/karma/commit/42bf787f87304e6be23dd3dac893b3c3f77d6764))
* **init:** generate config with the new syntax ([6b27fee5](http://github.com/karma-runner/karma/commit/6b27fee5a43a7d02e706355f62fe5105b4966c43))
* **reporter:** prevent throwing exception when null is sent to formatter ([3b49c385](http://github.com/karma-runner/karma/commit/3b49c385fcc8ef96e72be390df058bd278b40c17))
* **watcher:** ignore fs.stat errors ([74ccc9a8](http://github.com/karma-runner/karma/commit/74ccc9a8017f869bd7bbbf8831415964110a7073))


#### Features

* capture window.alert ([284c4f5c](http://github.com/karma-runner/karma/commit/284c4f5c9c481759fe564627a00d72ba5c54e433))
* ship html2js preprocessor as a default plugin ([37ecf416](http://github.com/karma-runner/karma/commit/37ecf41600a9b255ab3d57327cc83d64751642f5))
* fail if zero tests executed ([5670415e](http://github.com/karma-runner/karma/commit/5670415ecdc5e54902b479c78df5c3c422855e5c), closes [#468](http://github.com/karma-runner/karma/issues/468))
* **launcher:** normalize quoted paths ([f2155e0c](http://github.com/karma-runner/karma/commit/f2155e0c3305538c0fb95791e56f34743977a865), closes [#491](http://github.com/karma-runner/karma/issues/491))
* **web-server:** serve css files ([4e305545](http://github.com/karma-runner/karma/commit/4e305545ddf2726c1fe65c46efd5e7c1045ac041), closes [#431](http://github.com/karma-runner/karma/issues/431))

<a name="v0.9.4"></a>
### v0.9.4 (2013-06-28)


#### Bug Fixes

* **config:**
  * make the config changes backwards compatible ([593ad853](https://github.com/karma-runner/karma/commit/593ad853c330a7856f2112db2bfb288f67948fa6))
  * better errors if file invalid or does not exist ([74b533be](https://github.com/karma-runner/karma/commit/74b533beb34c115f5080d412a03573d269d540aa))
  * allow parsing the config multiple times ([78a7094e](https://github.com/karma-runner/karma/commit/78a7094e0f262c431e904f99cf356be53eee3510))
* **launcher:** better errors when loading launchers ([504e848c](https://github.com/karma-runner/karma/commit/504e848cf66b065380fa72e07f5337ae2d6e35b5))
* **preprocessor:**
  * do not show duplicate warnings ([47c641f7](https://github.com/karma-runner/karma/commit/47c641f7560d28e0d9eac7ae010566d296d5b628))
  * better errors when loading preprocessors ([3390a00b](https://github.com/karma-runner/karma/commit/3390a00b49c513a6da60f48044462118436130f8))
* **reporter:** better errors when loading reporters ([c645c060](https://github.com/karma-runner/karma/commit/c645c060c4f381902c2005eefe5b3a7bfa63cdcc))


#### Features

* **config:** pass the config object rather than a wrapper ([d2a3c854](https://github.com/karma-runner/karma/commit/d2a3c8546dc4b10bb9194047a1c11963639f3730))


#### Breaking Changes

* please update your karma.conf.js as follows ([d2a3c854](https://github.com/karma-runner/karma/commit/d2a3c8546dc4b10bb9194047a1c11963639f3730)):

```javascript
// before:
module.exports = function(karma) {
  karma.configure({port: 123});
  karma.defineLauncher('x', 'Chrome', {
    flags: ['--disable-web-security']
  });
  karma.definePreprocessor('y', 'coffee', {
    bare: false
  });
  karma.defineReporter('z', 'coverage', {
    type: 'html'
  });
};

// after:
module.exports = function(config) {
  config.set({
    port: 123,
    customLaunchers: {
      'x': {
        base: 'Chrome',
        flags: ['--disable-web-security']
      }
    },
    customPreprocessors: {
      'y': {
        base: 'coffee',
        bare: false
      }
    },
    customReporters: {
      'z': {
        base: 'coverage',
        type: 'html'
      }
    }
  });
};
```

<a name="v0.9.3"></a>
### v0.9.3 (2013-06-16)


#### Bug Fixes

* capturing console.log on IE ([fa4b686a](https://github.com/karma-runner/karma/commit/fa4b686a81ad826f256a4ca63c772af7ad6e411e), closes [#329](https://github.com/karma-runner/karma/issues/329))
* **config:** fix the warning when using old syntax ([5e55d797](https://github.com/karma-runner/karma/commit/5e55d797f7544a45c3042e301bbf71e8b830daf3))
* **init:** generate correct indentation ([5fc17957](https://github.com/karma-runner/karma/commit/5fc17957be761c06f6ae120c5d3ba800dba8d3a4))
* **launcher:**
  * ignore exit code when killing/timeouting ([1029bf2d](https://github.com/karma-runner/karma/commit/1029bf2d7d3d22986aa41439d2ce4115770f4dbd), closes [#444](https://github.com/karma-runner/karma/issues/444))
  * handle ENOENT error, do not retry ([7d790b29](https://github.com/karma-runner/karma/commit/7d790b29c09c1f3784fe648b7d5ed16add10b4ca), closes [#452](https://github.com/karma-runner/karma/issues/452))
* **logger:** configure the logger as soon as possible ([0607d67c](https://github.com/karma-runner/karma/commit/0607d67c15eab58ce83cce14ada70a1e2a9f17e9))
* **preprocessor:** use graceful-fs to prevent EACCESS errors ([279bcab5](https://github.com/karma-runner/karma/commit/279bcab54019a0f0af72c7c08017cf4cdefebe46), closes [#566](https://github.com/karma-runner/karma/issues/566))
* **watcher:** watch files that match watched directory ([39401175](https://github.com/karma-runner/karma/commit/394011753b918b8db807f31da9f5c316e296cf32), closes [#521](https://github.com/karma-runner/karma/issues/521))


#### Features

* simplify loading plugins using patterns like `karma-*` ([405a5a62](https://github.com/karma-runner/karma/commit/405a5a62d2ecc47a46b2ff069bfeb624f0b06982))
* **client:** capture all `console.*` log methods ([683e6dcb](https://github.com/karma-runner/karma/commit/683e6dcb9132de3caee39c809b5b58efe8236564))
* **config:**
  * make socket.io transports configurable ([bbd5eb86](https://github.com/karma-runner/karma/commit/bbd5eb8688b2bc1e3dd04910aa68fd19c5036b31))
  * allow configurable launchers, preprocessors, reporters ([76bdac16](https://github.com/karma-runner/karma/commit/76bdac1681f012749648f5a76b4a9d96c7a5ef20), closes [#317](https://github.com/karma-runner/karma/issues/317))
  * add warning if old constants are used ([7233c5fb](https://github.com/karma-runner/karma/commit/7233c5fb9e1c105032000bbcb9afaddf72ccbc97))
  * require config as a regular module ([a37fd6f7](https://github.com/karma-runner/karma/commit/a37fd6f7d28036b8da5fe98634cf711cebafc1ff), closes [#304](https://github.com/karma-runner/karma/issues/304))
* **helper:** improve useragent detection ([eb58768e](https://github.com/karma-runner/karma/commit/eb58768e32baf13b45d9649743d7ef45798ffb27))
* **init:**
  * generate coffee config files ([d2173717](https://github.com/karma-runner/karma/commit/d21737176c1d866a11249d626a75440b398171ce))
  * improve the questions a bit ([baecadb2](https://github.com/karma-runner/karma/commit/baecadb2f1a8f31c233edacafb1f8a4b736ea243))
* **proxy:** add https proxy support ([be878dc5](https://github.com/karma-runner/karma/commit/be878dc545a0dd266d5686387c976ce70f1a095c))


#### Breaking Changes

* Update your karma.conf.js to export a config function ([a37fd6f7](https://github.com/karma-runner/karma/commit/a37fd6f7d28036b8da5fe98634cf711cebafc1ff)):

```javascript
module.exports = function(karma) {
  karma.configure({
    autoWatch: true,
    // ...
  });
};
```

<a name="v0.9.2"></a>
### v0.9.2 (2013-04-16)


#### Bug Fixes

* better error reporting when loading plugins ([d9078a8e](https://github.com/karma-runner/karma/commit/d9078a8eca41df15f26b53e2375f722a48d0992d))
* **config:**
  * Separate ENOENT error handler from others ([e49dabe7](https://github.com/karma-runner/karma/commit/e49dabe783d6cfb2ee97b70ac01953e82f70f831))
  * ensure basePath is always resolved ([2e5c5aaa](https://github.com/karma-runner/karma/commit/2e5c5aaaddc4ad4e1ee9c8fa0388d3916827f403))


#### Features

* allow inlined plugins ([3034bcf9](https://github.com/karma-runner/karma/commit/3034bcf9b074b693afab9c62856346d6f305d0c0))
* **debug:** show skipped specs and failure details in the console ([42ab936b](https://github.com/karma-runner/karma/commit/42ab936b254983faa8ab0ee76a6278fb3aff7fa2))

<a name="v0.9.1"></a>
### v0.9.1 (2013-04-04)


#### Bug Fixes

* **init:** to not give false warning about missing requirejs ([562607a1](https://github.com/karma-runner/karma/commit/562607a16221b256c6e92ad2029154aac88eec8d))


#### Features

* ship coffee-preprocessor and requirejs as default plugins ([f34e30db](https://github.com/karma-runner/karma/commit/f34e30db4d25d484a30d12e3cb1c41069c0b263a))

<a name="v0.9.0"></a>
## v0.9.0 (2013-04-03)


#### Bug Fixes

* global error handler should propagate errors ([dec0c196](https://github.com/karma-runner/karma/commit/dec0c19651c251dcbc16c44a57775bcb37f78cf1), closes [#368](https://github.com/karma-runner/karma/issues/368))
* **config:**
  * Check if configFilePath is a string. Fixes #447. ([98724b6e](https://github.com/karma-runner/karma/commit/98724b6ef5a6ba60d487e7b774056832c6ca9d8c))
  * do not change urlRoot even if proxied ([8c138b50](https://github.com/karma-runner/karma/commit/8c138b504046a3aeb230b71e1049aa60ee46905d))
* **coverage:** always send a result object ([62c3c679](https://github.com/karma-runner/karma/commit/62c3c6790659f8f82f8a2ca5646aa424eeb28842), closes [#365](https://github.com/karma-runner/karma/issues/365))
* **init:**
  * generate plugins and frameworks config ([17798d55](https://github.com/karma-runner/karma/commit/17798d55988d61070f2b9f59574217208f2b497e))
  * fix for failing "testacular init" on Windows ([0b5b3853](https://github.com/karma-runner/karma/commit/0b5b385383f13ac8f29fa6e591a8634eefa04ab7))
* **preprocessor:** resolve relative patterns to basePath ([c608a9e5](https://github.com/karma-runner/karma/commit/c608a9e5a34a49da2971add8759a9422b74fa6fd), closes [#382](https://github.com/karma-runner/karma/issues/382))
* **runner:** send exit code as string ([ca75aafd](https://github.com/karma-runner/karma/commit/ca75aafdf6b7b425ee151c2ae4ede37933befe1f), closes [#403](https://github.com/karma-runner/karma/issues/403))


#### Features

* display the version when starting ([39617395](https://github.com/karma-runner/karma/commit/396173952addce3f6e904310686a42b102aa53f8), closes [#391](https://github.com/karma-runner/karma/issues/391))
* allow multiple preprocessors ([1d17c1aa](https://github.com/karma-runner/karma/commit/1d17c1aacf607d6c4269f05df97d024bc9ca994e))
* allow plugins ([125ab4f8](https://github.com/karma-runner/karma/commit/125ab4f88a7cf49fd7df32264a9847847e2326ca))
* **config:**
  * always ignore the config file itself ([103bc0f8](https://github.com/karma-runner/karma/commit/103bc0f878a8870770c8a8afce0a3fbf8a516ea7))
  * normalize string preprocessors into an array ([4dde1608](https://github.com/karma-runner/karma/commit/4dde16087d0a704a47528d44e23ace0c536d8c72))
* **web-server:** allow custom file handlers and mime types ([2df88287](https://github.com/karma-runner/karma/commit/2df8828742041fd09c0b45d6a62ebd7552116589))


#### Breaking Changes

* reporters, launchers, preprocessors, adapters are separate plugins now, in order to use them, you need to install the npm package (probably add it as a `devDependency` into your `package.json`) and load in the `karma.conf.js` with `plugins = ['karma-jasmine', ...]`. Karma ships with couple of default plugins (karma-jasmine, karma-chrome-launcher, karma-phantomjs-launcher).

* frameworks (such as jasmine, mocha, qunit) are configured using `frameworks = ['jasmine'];` instead of prepending `JASMINE_ADAPTER` into files.


<a name="v0.8.0"></a>
## v0.8.0 (2013-03-18)


#### Breaking Changes

* rename the project to "Karma":
- whenever you call the "testacular" binary, change it to "karma", eg. `testacular start` becomes `karma start`.
- if you rely on default name of the config file, change it to `karma.conf.js`.
- if you access `__testacular__` object in the client code, change it to `__karma__`, eg. `window.__testacular__.files` becomes `window.__karma__.files`. ([026a20f7](https://github.com/karma-runner/karma/commit/026a20f7b467eb3b39c68ed509acc06e5dad58e6))

<a name="v0.6.1"></a>
### v0.6.1 (2013-03-18)


#### Bug Fixes

* **config:** do not change urlRoot even if proxied ([1be1ae1d](https://github.com/karma-runner/karma/commit/1be1ae1dc7ff7314f4ac2854815cb39d31362f14))
* **coverage:** always send a result object ([2d210aa6](https://github.com/karma-runner/karma/commit/2d210aa6697991f2eba05de58a696c5210485c88), closes [#365](https://github.com/karma-runner/karma/issues/365))
* **reporter.teamcity:** report spec names and proper browser name ([c8f6f5ea](https://github.com/karma-runner/karma/commit/c8f6f5ea0c5c40d37b511d51b49bd22c9da5ea86))

<a name="v0.6.0"></a>
## v0.6.0 (2013-02-22)

<a name="v0.5.11"></a>
### v0.5.11 (2013-02-21)


#### Bug Fixes

* **adapter.requirejs:** do not configure baseUrl automatically ([63f3f409](https://github.com/karma-runner/karma/commit/63f3f409ae85a5137396a7ed6537bedfe4437cb3), closes [#291](https://github.com/karma-runner/karma/issues/291))
* **init:** add missing browsers (Opera, IE) ([f39e5645](https://github.com/karma-runner/karma/commit/f39e5645ec561c2681d907f7c1611f01911ee8fd))
* **reporter.junit:** Add browser log output to JUnit.xml ([f108799a](https://github.com/karma-runner/karma/commit/f108799a4d8fd95b8c0250ee83c23ada25d026b9), closes [#302](https://github.com/karma-runner/karma/issues/302))


#### Features

* add Teamcity reporter ([03e700ae](https://github.com/karma-runner/karma/commit/03e700ae2234ca7ddb8f9235343e3b0c80868bbd))
* **adapter.jasmine:** remove only last failed specs anti-feature ([435bf72c](https://github.com/karma-runner/karma/commit/435bf72cb12112462940c8114fbaa19f9de38531), closes [#148](https://github.com/karma-runner/karma/issues/148))
* **config:** allow empty config file when called programmatically ([f3d77424](https://github.com/karma-runner/karma/commit/f3d77424009f621e1fb9d60eeec7f052ebb3c585), closes [#358](https://github.com/karma-runner/karma/issues/358))

<a name="v0.5.10"></a>
### v0.5.10 (2013-02-14)


#### Bug Fixes

* **init:** fix the logger configuration ([481dc3fd](https://github.com/karma-runner/karma/commit/481dc3fd75f45a0efa8aabdb1c71e8234b9e8a06), closes [#340](https://github.com/karma-runner/karma/issues/340))
* **proxy:** fix crashing proxy when browser hangs connection ([1c78a01a](https://github.com/karma-runner/karma/commit/1c78a01a19411accb86f0bde9e040e5088752575))


#### Features

* set urlRoot to /__karma__/ when proxying the root ([8b4fd64d](https://github.com/karma-runner/karma/commit/8b4fd64df6b7d07b5479e43dcd8cd2aa5e1efc9c))
* **adapter.requirejs:** normalize paths before appending timestamp ([94889e7d](https://github.com/karma-runner/karma/commit/94889e7d2de701c67a2612e3fc6a51bfae891d36))
* update dependencies to the latest ([93f96278](https://github.com/karma-runner/karma/commit/93f9627817f2d5d9446de9935930ca85cfa7df7f), [e34d8834](https://github.com/karma-runner/karma/commit/e34d8834d69ec4e022fcd6e1be4055add96d693c))


<a name="v0.5.9"></a>
### v0.5.9 (2013-02-06)


#### Bug Fixes

* **adapter.requirejs:** show error if no timestamp defined for a file ([59dbdbd1](https://github.com/karma-runner/karma/commit/59dbdbd136baa87467b9b9a4cb6ce226ae87bbef))
* **init:** fix logger configuration ([557922d7](https://github.com/karma-runner/karma/commit/557922d71941e0929f9cdc0d3794424a1f27b311))
* **reporter:** remove newline from base reporter browser dump ([dfae18b6](https://github.com/karma-runner/karma/commit/dfae18b63b413a1e6240d00b9dc0521ac0386ec5), closes [#297](https://github.com/karma-runner/karma/issues/297))
* **reporter.dots:** only add newline to message when needed ([dbe1155c](https://github.com/karma-runner/karma/commit/dbe1155cb57fc4caa792f83f45288238db0fc7e0)

#### Features

* add "debug" button to easily open debugging window ([da85aab9](https://github.com/karma-runner/karma/commit/da85aab927edd1614e4e05b136dee834344aa3cb))
* **config:** support running on a custom hostname ([b8c5fe85](https://github.com/karma-runner/karma/commit/b8c5fe8533b13fd59cbf48972d2021069a84ae5b))
* **reporter.junit:** add a 'skipped' tag for skipped testcases ([6286406e](https://github.com/karma-runner/karma/commit/6286406e0a36a61125ea16d6f49be07030164cb0), closes [#321](https://github.com/karma-runner/karma/issues/321))


### v0.5.8
* Fix #283
* Suppress global leak for istanbul
* Fix growl reporter to work with `testacular run`
* Upgrade jasmine to 1.3.1
* Fix file sorting
* Fix #265
* Support for more mime-types on served static files
* Fix opening Chrome on Windows
* Upgrade growly to 1.1.0

### v0.5.7
* Support code coverage for qunit.
* Rename port-runner option in cli to runner-port
* Fix proxy handler (when no proxy defined)
* Fix #65

### v0.5.6
* Growl reporter !
* Batch changes (eg. `git checkout` causes only single run now)
* Handle uncaught errors and disconnect all browsers
* Global binary prefers local versions

### v0.5.5
* Add QUnit adapter
* Report console.log()

### v0.5.4
* Fix PhantomJS launcher
* Fix html2js preprocessor
* NG scenario adapter: show html output

### v0.5.3
* Add code coverage !

### v0.5.2
* Init: ask about using Require.js

### v0.5.1
* Support for Require.js
* Fix testacular init basePath

## v0.5.0
* Add preprocessor for LiveScript
* Fix JUnit reporter
* Enable process global in config file
* Add OS name in the browser name
* NG scenario adapter: hide other outputs to make it faster
* Allow config to be written in CoffeeScript
* Allow espaced characters in served urls

## v0.4.0 (stable)

### v0.3.12
* Allow calling run() pragmatically from JS

### v0.3.11
* Fix runner to wait for stdout, stderr
* Make routing proxy always changeOrigin

### v0.3.10
* Fix angular-scenario adapter + junit reporter
* Use flash socket if web socket not available

### v0.3.9
* Retry starting a browser if it does not capture
* Update mocha to 1.5.0
* Handle mocha's xit

### v0.3.8
* Kill browsers that don't capture in captureTimeout ms
* Abort build if any browser fails to capture
* Allow multiple profiles of Firefox

### v0.3.7
* Remove Travis hack
* Fix Safari launcher

### v0.3.6
* Remove custom launcher (constructor)
* Launcher - use random id to allow multiple instances of the same browser
* Fix Firefox launcher (creating profile)
* Fix killing browsers on Linux and Windows

### v0.3.5
* Fix opera launcher to create new prefs with disabling all pop-ups

### v0.3.4
* Change "reporter" config to "reporters"
* Allow multiple reporters
* Fix angular-scenario adapter to report proper description
* Add JUnit xml reporter
* Fix loading files from multiple drives on Windows
* Fix angular-scenario adapter to report total number of tests

### v0.3.3
* Allow proxying files, not only directories

### v0.3.2
* Disable autoWatch if singleRun
* Add custom script browser launcher
* Fix cleaning temp folders

### v0.3.1
* Run tests on start (if watching enabled)
* Add launcher for IE8, IE9

## v0.3.0
* Change browser binaries on linux to relative
* Add report-slower-than to CLI options
* Fix PhantomJS binary on Travis CI

## v0.2.0 (stable)

### v0.1.3
* Launch Canary with crankshaft disabled
* Make the captured page nicer

### v0.1.2
* Fix jasmine memory leaks
* support __filename and __dirname in config files

### v0.1.1
* Report slow tests (add `reportSlowerThan` config option)
* Report time in minutes if it's over 60 seconds
* Mocha adapter: add ability to fail during beforeEach/afterEach hooks
* Mocha adapter: add dump()
* NG scenario adapter: failure includes step name
* Redirect /urlRoot to /urlRoot/
* Fix serving with urlRoot

## v0.1.0
* Adapter for AngularJS scenario runner
* Allow serving Testacular from a subpath
* Fix race condition in testacular run
* Make testacular one binary (remove `testacular-run`, use `testacular run`)
* Add support for proxies
* Init script for generating config files (`testacular init`)
* Start Firefox without custom profile if it fails
* Preserve order of watched paths for easier debugging
* Change default port to 9876
* Require node v0.8.4+

### v0.0.17
* Fix race condition in manually triggered run
* Fix autoWatch config

### v0.0.16
* Mocha adapter
* Fix watching/resolving on Windows
* Allow glob patterns
* Watch new files
* Watch removed files
* Remove unused config (autoWatchInterval)

### v0.0.15
* Remove absolute paths from urls (fixes Windows issue with C:\\)
* Add browser launcher for PhantomJS
* Fix some more windows issues

### v0.0.14
* Allow require() inside config file
* Allow custom browser launcher
* Add browser launcher for Opera, Safari
* Ignore signals on windows (not supported yet)

### v0.0.13
* Single run mode (capture browsers, run tests, exit)
* Start browser automatically (chrome, canary, firefox)
* Allow loading external files (urls)

### v0.0.12
* Allow console in config
* Warning if pattern does not match any file

### v0.0.11
* Add timing (total / net - per specs)
* Dots reporter - wrap at 80

### v0.0.10
* Add DOTS reporter
* Add no-colors option for reporters
* Fix web server to expose only specified files

### v0.0.9
* Proper exit code for runner
* Dynamic port asigning (if port already in use)
* Add log-leve, log-colors cli arguments + better --help
* Fix some IE errors (indexOf, forEach fallbacks)

### v0.0.8
* Allow overriding configuration by cli arguments (+ --version, --help)
* Persuade IE8 to not cache context.html
* Exit runner if no captured browser
* Fix delayed execution (streaming to runner)
* Complete run if browser disconnects
* Ignore results from previous run (after server reconnecting)
* Server disconnects - cancel execution, clear browser info

### v0.0.7
* Rename to Testacular

### v0.0.6
* Better debug mode (no caching, no timestamps)
* Make dump() a bit better
* Disconnect browsers on SIGTERM (kill, killall default)

### v0.0.5
* Fix memory (some :-D) leaks
* Add dump support
* Add runner.html

### v0.0.4
* Progress bar reporting
* Improve error formatting
* Add Jasmine lib (with iit, ddescribe)
* Reconnect client each 2sec, remove exponential growing

### v0.0.3
* Jasmine adapter: ignore last failed filter in exclusive mode
* Jasmine adapter: add build (no global space pollution)

### 0.0.2
* Run only last failed tests (jasmine adapter)

### 0.0.1
* Initial version with only very basic features
