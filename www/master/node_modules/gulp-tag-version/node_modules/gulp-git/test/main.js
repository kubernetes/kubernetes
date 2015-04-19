var fs = require('fs');
var path = require('path');
var should = require('should');
var gutil = require('gulp-util');
var rimraf = require('rimraf');
var git = require('../');

require('mocha');

var testFile = __dirname + '/test.js';
fs.openSync(testFile, 'w');

var testCommit = path.join(__dirname, '/.git/COMMIT_EDITMSG');

describe('gulp-git', function() {

  describe('normal usage ', function(){

    it('should initialize a empty git repo', function(done) {
      git.init({cwd:"./test/"}, function(){
        should.exist("test/.git/");
        done();
      });
    });

    it('should add files to the git repo', function(done) {
      var fakeFile = new gutil.File({
        base: 'test/',
        cwd: 'test/',
        path: path.join(__dirname, '/test.js'),
        contents: new Buffer(fs.readFileSync('test/test.js'))
      });
      var gitS = git.add();
      gitS.once('data', function(newFile){
        should.exist(newFile);
        should.exist('test/.git/objects/');
        done();
      });
      gitS.write(fakeFile);
    });

    it('should add a Remote to the git repo', function(done) {
      git.addRemote('origin', 'https://github.com/stevelacy/git-test', {cwd:"./test/"} , function(){
        should.exist('test/.git/');
        String(fs.readFileSync('test/.git/config').toString('utf8')).should.match(/https:\/\/github.com\/stevelacy\/git-test/);
        done();
      });
    });

/*
    it('should pull from the remote repo', function(done) {
      git.pull('origin', 'master', {cwd: "./test/"}, function(){
        should.exist('./test/.git/refs/heads/master');
        done();
      });
    });
  
  // This must be run on a system which has git installed, and has git configured.
*/
    it('should tag a version of the repo', function(done) {
      
      git.tag('v1.2.3', 'message', {cwd: "./test/"}, function(){
        should.exist('test/.git/refs/tags/v1.2.3');
        done();
      });

    });

    it('should commit a file to the repo', function(done) {
      var fakeFile = new gutil.File({
        base: 'test/',
        cwd: 'test/',
        path: path.join(__dirname, 'test.js'),
        contents: new Buffer(fs.readFileSync('./test/test.js'))
      });
      var gitS = git.commit('initial commit');
      gitS.once('data', function(newFile){
        setTimeout(function(){
          String(fs.readFileSync(testCommit).toString('utf8')).should.match(/initial commit/);
        }, 100);
        done();
      });
      gitS.write(fakeFile);
    });

    it('should create a new branch', function(done){
      git.branch("testBranch", {cwd: "./test/"}, function(){
        should.exist('test/.git/refs/heads/testBranch');
        done();
      });
    });

    it('should merge branches', function(done){
      git.merge("testBranch", {cwd: "./test/"}, function(){
        setTimeout(function(){
          String(fs.readFileSync(testCommit).toString('utf8')).should.match(/initial commit/);
        }, 100);
        done();
      });
    });

    it('should checkout a branch', function(done) {
      var fakeFile = new gutil.File({
        base: 'test/',
        cwd: 'test/',
        path: path.join(__dirname, 'test.js')
      });
      var gitS = git.checkout("testBranch", '-b');
      gitS.once('data', function () {
        String(fs.readFileSync('test/.git/HEAD').toString('utf8')).should.match(/ref\: refs\/heads\/master/);
        done();
      });
      gitS.write(fakeFile);
    });



  // With options



    it('should add files to the git repo - with options', function(done) {
      var fakeFile = new gutil.File({
        base: 'test/',
        cwd: 'test/',
        path: path.join(__dirname, '/test.js'),
        contents: new Buffer(fs.readFileSync('test/test.js'))
      });
      var gitS = git.add({args: "-u"});
      gitS.once('data', function(newFile){
        should.exist(newFile);
        should.exist('test/.git/objects/');
        done();
      });
      gitS.write(fakeFile);
    });

    it('should add a Remote to the git repo - with options', function(done) {

      git.addRemote('origin', 'https://github.com/stevelacy/git-test', {cwd:"./test/", args: "--no-tag"} , function(){
        should.exist('test/.git/');
          String(fs.readFileSync('test/.git/config').toString('utf8')).should.match(/https:\/\/github.com\/stevelacy\/git-test/);
          done();
      });

    });
/*
    it('should pull from the remote repo - with options', function(done) {
      git.pull('origin', 'master', {cwd: "./test/", args: "-v"}, function(){
        should.exist('./test/.git/refs/heads/master');
        done();
      });
    });
*/

    it('should commit a file to the repo - with options', function(done) {

      var fakeFile = new gutil.File({
        base: 'test/',
        cwd: 'test/',
        path: path.join(__dirname, 'test.js'),
        contents: new Buffer(fs.readFileSync('./test/test.js'))
      });
      var gitS = git.commit('initial commit', {args: "-v"});
      gitS.once('data', function(newFile){
        setTimeout(function(){
          String(fs.readFileSync(testCommit).toString('utf8')).should.match(/initial commit/);
        }, 100);
        done();
      });
      gitS.write(fakeFile);

    });


    it('should create a new branch - with options', function(done){
      git.branch("testBranch", {cwd: "./test/", args: "--track"}, function(){
        should.exist('test/.git/refs/heads/testBranch');
        done();
      });
    });
    
    it('should merge branches', function(done){
      git.merge("testBranch", {cwd: "./test/"}, function(){
        setTimeout(function(){
          String(fs.readFileSync(testCommit).toString('utf8')).should.match(/initial commit/);
        }, 100);
        done();
      });
    });


    it('should rm a file', function(done) {
      var fakeFile = new gutil.File({
        base: 'test/',
        cwd: 'test/',
        path: testFile
      });
      var gitS = git.rm();
      gitS.once('data', function (newFile) {
        setTimeout(function(){
          fs.exists(testFile, function(exists) {
            exists.should.be.false;
          });
        }, 100);
        done();
      });
      gitS.write(fakeFile);
    });

  });

  after(function(done){
    rimraf('test/.git', function(err){
      if(err) return err;
      done();
    });
  });

});
