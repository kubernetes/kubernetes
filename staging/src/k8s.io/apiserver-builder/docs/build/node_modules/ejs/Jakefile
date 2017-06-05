var fs = require('fs');
var execSync = require('child_process').execSync;
var exec = function (cmd) {
  execSync(cmd, {stdio: 'inherit'});
};

/* global jake, task, desc, publishTask */

task('build', ['lint', 'clean', 'browserify', 'minify'], function () {
  console.log('Build completed.');
});

desc('Cleans browerified/minified files and package files');
task('clean', ['clobber'], function () {
  jake.rmRf('./ejs.js');
  jake.rmRf('./ejs.min.js');
  console.log('Cleaned up compiled files.');
});

desc('Lints the source code');
task('lint', function () {
  exec('./node_modules/.bin/eslint \"**/*.js\" Jakefile');
  console.log('Linting completed.');
});

task('browserify', function () {
  exec('./node_modules/browserify/bin/cmd.js --standalone ejs lib/ejs.js > ejs.js');
  console.log('Browserification completed.');
});

task('minify', function () {
  exec('./node_modules/uglify-js/bin/uglifyjs ejs.js > ejs.min.js');
  console.log('Minification completed.');
});

task('doc', function (dev) {
  jake.rmRf('out');
  var p = dev ? '-p' : '';
  exec('./node_modules/.bin/jsdoc ' + p + ' -c jsdoc.json lib/* docs/jsdoc/*');
  console.log('Documentation generated.');
});

task('docPublish', ['doc'], function () {
  fs.writeFileSync('out/CNAME', 'api.ejs.co');
  console.log('Pushing docs to gh-pages...');
  exec('./node_modules/.bin/git-directory-deploy --directory out/');
  console.log('Docs published to gh-pages.');
});

task('test', ['lint'], function () {
  exec('./node_modules/.bin/mocha');
});

publishTask('ejs', ['build'], function () {
  this.packageFiles.include([
    'Jakefile',
    'README.md',
    'LICENSE',
    'package.json',
    'ejs.js',
    'ejs.min.js',
    'lib/**',
    'test/**'
  ]);
});

jake.Task.publish.on('complete', function () {
  console.log('Updating hosted docs...');
  console.log('If this fails, run jake docPublish to re-try.');
  jake.Task.docPublish.invoke();
});
