'use strict';

module.exports = function(grunt) {

  // Report the elapsed execution time of tasks.
  require('time-grunt')(grunt);

  // Project configuration.
  grunt.initConfig({

    // Metadata required for build.
    build: grunt.file.readYAML('build/build.yml'),
    pkg: grunt.file.readJSON('package.json'),
    meta: {
      license: '<%= _.pluck(pkg.licenses, "type").join(", ") %>',
      copyright: 'Copyright (c) 2009-<%= grunt.template.today("yyyy") %>',
      banner:
        '/*!\n' +
        ' * Less - <%= pkg.description %> v<%= pkg.version %>\n' +
        ' * http://lesscss.org\n' +
        ' *\n' +
        ' * <%= meta.copyright %>, <%= pkg.author.name %> <<%= pkg.author.email %>>\n' +
        ' * Licensed under the <%= meta.license %> License.\n' +
        ' *\n' +
        ' */\n\n' +
        ' /**' +
        ' * @license <%= meta.license %>\n' +
        ' */\n\n'
    },

    shell: {
      options: {stdout: true, failOnError: true},
      test: {
        command: 'node test'
      },
      benchmark: {
        command: 'node benchmark/less-benchmark.js'
      },
      "browsertest-server": {
          command: 'node node_modules/http-server/bin/http-server . -p 8088'
      },
      "sourcemap-test": {
        command: [
            'node bin/lessc --source-map --source-map-map-inline test/less/import.less test/sourcemaps/import.css',
            'node bin/lessc --source-map --source-map-map-inline test/less/sourcemaps/basic.less test/sourcemaps/basic.css',
            'node node_modules/http-server/bin/http-server test/sourcemaps -p 8084'].join('&&')
      }
    },
    concat: {
      options: {
        stripBanners: 'all',
        banner: '<%= meta.banner %>\n\n(function (window, undefined) {',
        footer: '\n})(window);'
      },
      // Browser versions
      browsertest: {
        src: ['<%= build.browser %>'],
        dest: 'test/browser/less.js'
      },
      stable: {
        src: ['<%= build.browser %>'],
        dest: 'dist/less-<%= pkg.version %>.js'
      },
      // Rhino
      rhino: {
        options: {
          banner: '/* Less.js v<%= pkg.version %> RHINO | <%= meta.copyright %>, <%= pkg.author.name %> <<%= pkg.author.email %>> */\n\n',
          footer: '' // override task-level footer
        },
        src: ['<%= build.rhino %>'],
        dest: 'dist/less-rhino-<%= pkg.version %>.js'
      },
      // lessc for Rhino
      rhinolessc: {
        options: {
          banner: '/* Less.js v<%= pkg.version %> RHINO | <%= meta.copyright %>, <%= pkg.author.name %> <<%= pkg.author.email %>> */\n\n',
          footer: '' // override task-level footer
        },
        src: ['<%= build.rhinolessc %>'],
        dest: 'dist/lessc-rhino-<%= pkg.version %>.js'
      }
    },

    uglify: {
      options: {
        banner: '<%= meta.banner %>',
        mangle: true
      },
      stable: {
        src: ['<%= concat.stable.dest %>'],
        dest: 'dist/less-<%= pkg.version %>.min.js'
      }
    },

    jshint: {
      options: {jshintrc: '.jshintrc'},
      files: {
        src: [
          'Gruntfile.js',
          'lib/less/**/*.js'
        ]
      }
    },

    connect: {
      server: {
        options: {
          port: 8081
        }
      }
    },

    jasmine: {
      options: {
        // version: '2.0.0-rc2',
        keepRunner: true,
        host: 'http://localhost:8081/',
        vendor: ['test/browser/common.js', 'test/browser/less.js'],
        template: 'test/browser/test-runner-template.tmpl'
      },
      main: {
        // src is used to build list of less files to compile
        src: ['test/less/*.less', '!test/less/javascript.less', '!test/less/urls.less', '!test/less/empty.less'],
        options: {
          helpers: 'test/browser/runner-main-options.js',
          specs: 'test/browser/runner-main-spec.js',
          outfile: 'tmp/browser/test-runner-main.html'
        }
      },
      legacy: {
        src: ['test/less/legacy/*.less'],
        options: {
          helpers: 'test/browser/runner-legacy-options.js',
          specs: 'test/browser/runner-legacy-spec.js',
          outfile: 'tmp/browser/test-runner-legacy.html'
        }
      },
      errors: {
        src: ['test/less/errors/*.less', '!test/less/errors/javascript-error.less'],
        options: {
          timeout: 20000,
          helpers: 'test/browser/runner-errors-options.js',
          specs: 'test/browser/runner-errors-spec.js',
          outfile: 'tmp/browser/test-runner-errors.html'
        }
      },
      noJsErrors: {
        src: ['test/less/no-js-errors/*.less'],
        options: {
          helpers: 'test/browser/runner-no-js-errors-options.js',
          specs: 'test/browser/runner-no-js-errors-spec.js',
          outfile: 'tmp/browser/test-runner-no-js-errors.html'
        }
      },
      browser: {
        src: ['test/browser/less/*.less'],
        options: {
          helpers: 'test/browser/runner-browser-options.js',
          specs: 'test/browser/runner-browser-spec.js',
          outfile: 'tmp/browser/test-runner-browser.html'
        }
      },
      relativeUrls: {
        src: ['test/browser/less/relative-urls/*.less'],
        options: {
          helpers: 'test/browser/runner-relative-urls-options.js',
          specs: 'test/browser/runner-relative-urls-spec.js',
          outfile: 'tmp/browser/test-runner-relative-urls.html'
        }
      },
      rootpath: {
        src: ['test/browser/less/rootpath/*.less'],
        options: {
          helpers: 'test/browser/runner-rootpath-options.js',
          specs: 'test/browser/runner-rootpath-spec.js',
          outfile: 'tmp/browser/test-runner-rootpath.html'
        }
      },
      rootpathRelative: {
        src: ['test/browser/less/rootpath-relative/*.less'],
        options: {
          helpers: 'test/browser/runner-rootpath-relative-options.js',
          specs: 'test/browser/runner-rootpath-relative-spec.js',
          outfile: 'tmp/browser/test-runner-rootpath-relative.html'
        }
      },
      production: {
        src: ['test/browser/less/production/*.less'],
        options: {
          helpers: 'test/browser/runner-production-options.js',
          specs: 'test/browser/runner-production-spec.js',
          outfile: 'tmp/browser/test-runner-production.html'
        }
      },
      modifyVars: {
        src: ['test/browser/less/modify-vars/*.less'],
        options: {
          helpers: 'test/browser/runner-modify-vars-options.js',
          specs: 'test/browser/runner-modify-vars-spec.js',
          outfile: 'tmp/browser/test-runner-modify-vars.html'
        }
      },
      globalVars: {
        src: ['test/browser/less/global-vars/*.less'],
        options: {
          helpers: 'test/browser/runner-global-vars-options.js',
          specs: 'test/browser/runner-global-vars-spec.js',
          outfile: 'tmp/browser/test-runner-global-vars.html'
        }
      },
      postProcessor: {
        src: ['test/browser/less/postProcessor/*.less'],
        options: {
          helpers: 'test/browser/runner-postProcessor-options.js',
          specs: 'test/browser/runner-postProcessor.js',
          outfile: 'tmp/browser/test-postProcessor.html'
        }
      }
    },

    // Clean the version of less built for the tests
    clean: {
      test: ['test/browser/less.js', 'tmp'],
      "sourcemap-test": ['test/sourcemaps/*.css', 'test/sourcemaps/*.map']
    }
  });

  // Load these plugins to provide the necessary tasks
  require('matchdep').filterDev('grunt-*').forEach(grunt.loadNpmTasks);

  // Actually load this plugin's task(s).
  grunt.loadTasks('build/tasks');

  // by default, run tests
  grunt.registerTask('default', [
    'test'
  ]);

  // Release
  grunt.registerTask('stable', [
    'concat:stable',
    'uglify:stable'
  ]);

  // Release Rhino Version
  grunt.registerTask('rhino', [
    'concat:rhino',
    'concat:rhinolessc'
  ]);

  // Run all browser tests
  grunt.registerTask('browsertest', [
    'browser',
    'connect',
    'jasmine'
  ]);

  // setup a web server to run the browser tests in a browser rather than phantom
  grunt.registerTask('browsertest-server', [
    'shell:browsertest-server'
  ]);

  // Create the browser version of less.js
  grunt.registerTask('browser', [
    'concat:browsertest'
  ]);

  // Run all tests
  grunt.registerTask('test', [
    'clean',
    'jshint',
    'shell:test',
    'browsertest'
  ]);

  // generate a good test environment for testing sourcemaps
  grunt.registerTask('sourcemap-test', [
    'clean:sourcemap-test',
    'shell:sourcemap-test'
  ]);

  // Run benchmark
  grunt.registerTask('benchmark', [
    'shell:benchmark'
  ]);
};
