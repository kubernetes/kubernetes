var assert = require('assert');

describe('NPM Config on package.json', function () {
    describe('Setting process.env.npm_package_config', function () {
        /*jshint camelcase:false*/
        process.env.npm_package_config_bower_directory = 'npm-path';
        process.env.npm_package_config_bower_colors = false;

        var config = require('../lib/Config').create().load()._config;

        it('should return "npm-path" for "bower_directory"', function () {
            assert.equal('npm-path', config.directory);
        });
        it('should return "false" for "bower_colors"', function () {
            assert.equal('false', config.colors);
        });
    });
});

require('./util/index');
