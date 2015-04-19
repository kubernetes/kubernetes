module.exports = function(grunt) {
  var pkg = require("./package.json");
  global.jasmineVersion = pkg.version;
  var versionString = 'v' + pkg.version;

  grunt.initConfig({
    pkg: pkg,
    jshint: {all: ['lib/**/*.js', 'spec/**/*.js']}
  });

  var shell = require('shelljs');
  function runCommands(commands, done) {
    var command = commands.shift();

    if (command) {
      shell.exec(command, function(exitCode) {
        if (exitCode !== 0) {
          grunt.fail.fatal("Command `" + command + "` failed", exitCode);
          done();
        } else {
          runCommands(commands, done);
        }
      });
    } else {
      done();
    }
  }

  // depend on jshint:all, specs?
  grunt.registerTask('release',
                     'Create tag ' + versionString + ' and push jasmine-' + pkg.version + ' to NPM',
                     function() {
    var done = this.async(),
        commands = ['git tag ' + versionString, 'git push origin master --tags', 'npm publish'];

    runCommands(commands, done);
  });

  grunt.loadNpmTasks('grunt-contrib-jshint');

  grunt.loadTasks('tasks');

  grunt.registerTask('default', ['jshint:all', 'specs']);
};
