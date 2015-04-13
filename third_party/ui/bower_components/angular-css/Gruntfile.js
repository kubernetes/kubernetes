'use strict';

module.exports = function(grunt) {
    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),

        karma: {
            unit: {
                options: {
                    files: [
                        'node_modules/angular/angular.js',
                        'node_modules/angular-mocks/angular-mocks.js',
                        'node_modules/chai/chai.js',
                        'angular-css.js',
                        'test/spec.js'
                    ]
                },

                frameworks: ['mocha'],

                browsers: [
                    'Chrome',
                    'PhantomJS',
                    'Firefox'
                ],

                singleRun: true
            }
        },

        uglify: {
            options: {
                banner: '/*! <%= pkg.name %> <%= pkg.version %> | Copyright (c) <%= grunt.template.today("yyyy") %> DOOR3, Alex Castillo | MIT License */'
            },

            build: {
                src: '<%= pkg.name %>.js',
                dest: '<%= pkg.name %>.min.js'
            }
        }
    });

    grunt.loadNpmTasks('grunt-contrib-uglify');
    grunt.loadNpmTasks('grunt-karma');

    grunt.registerTask('test', ['karma']);

    grunt.registerTask('default', [
        'test',
        'uglify'
    ]);
};
