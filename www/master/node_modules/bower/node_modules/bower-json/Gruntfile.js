module.exports = function (grunt) {

    'use strict';

    grunt.loadNpmTasks('grunt-contrib-jshint');
    grunt.loadNpmTasks('grunt-contrib-watch');
    grunt.loadNpmTasks('grunt-simple-mocha');

    // Project configuration.
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
                reporter: 'spec'
            },
            full: { src: ['test/test.js'] },
            short: {
                options: {
                    reporter: 'dot'
                },
                src: ['test/test.js']
            },
            build: {
                options: {
                    reporter: 'tap'
                },
                src: ['test/test.js']
            }
        },


        watch: {
            files: ['<%= jshint.files %>'],
            tasks: ['jshint', 'simplemocha:short']
        }

    });

    // Default task.
    grunt.registerTask('test', ['simplemocha:full']);
    grunt.registerTask('default', ['jshint', 'test']);
};
