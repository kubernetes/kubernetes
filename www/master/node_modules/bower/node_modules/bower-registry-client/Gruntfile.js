'use strict';
module.exports = function (grunt) {
    grunt.loadNpmTasks('grunt-contrib-jshint');
    grunt.loadNpmTasks('grunt-contrib-watch');
    grunt.loadNpmTasks('grunt-simple-mocha');

    grunt.initConfig({
        jshint: {
            files: [
                'Gruntfile.js',
                'lib/**/*.js',
                'test/**/*.js'
            ],
            options: {
                jshintrc: '.jshintrc'
            }
        },
        simplemocha: {
            options: {
                reporter: 'spec',
                timeout: 20000
            },
            full: {
                src: ['test/runner.js']
            },
            short: {
                options: {
                    reporter: 'dot'
                },
                src: ['test/runner.js']
            },
            build: {
                options: {
                    reporter: 'tap'
                },
                src: ['test/runner.js']
            }
        },
        watch: {
            files: ['<%= jshint.files %>'],
            tasks: ['jshint', 'simplemocha:short']
        }
    });

    grunt.registerTask('test', ['simplemocha:full']);
    grunt.registerTask('default', ['jshint', 'test']);
};
