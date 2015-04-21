var gulp = require('gulp'), concat = require('gulp-concat'), uglify = require('gulp-uglify'),
    // jade = require('gulp-jade'),
    less = require('gulp-less'), path = require('path'),
    livereload = require('gulp-livereload'),  // Livereload plugin needed:
    // https://chrome.google.com/webstore/detail/livereload/jnihajbhpnppcggbcgedagnkighmdlei
    // marked = require('marked'), // For :markdown filter in jade
    path = require('path'), changed = require('gulp-changed'), prettify = require('gulp-html-prettify'),
    w3cjs = require('gulp-w3cjs'), rename = require('gulp-rename'),
    // flip = require('css-flip'),
    through = require('through2'), gutil = require('gulp-util'), htmlify = require('gulp-angular-htmlify'),
    minifyCSS = require('gulp-minify-css'), gulpFilter = require('gulp-filter'), expect = require('gulp-expect-file'),
    gulpsync = require('gulp-sync')(gulp), ngAnnotate = require('gulp-ng-annotate'),
    sourcemaps = require('gulp-sourcemaps'), del = require('del'), jsoncombine = require('gulp-jsoncombine'),
    ngConstant = require('gulp-ng-constant'), argv = require('yargs').argv, foreach = require('gulp-foreach'),
    gcallback = require('gulp-callback'), changeCase = require('change-case'),
    tag_version = require('gulp-tag-version'), PluginError = gutil.PluginError;

// LiveReload port. Change it only if there's a conflict
var lvr_port = 35729;

var W3C_OPTIONS = {
  // Set here your local validator if your using one. leave it empty if not
  // uri: 'http://validator/check',
  doctype: 'HTML5',
  output: 'json',
  // Remove some messages that angular will always display.
  filter: function(message) {
    if (/Element head is missing a required instance of child element title/.test(message)) return false;
    if (/Attribute .+ not allowed on element .+ at this point/.test(message)) return false;
    if (/Element .+ not allowed as child of element .+ in this context/.test(message)) return false;
    if (/Comments seen before doctype./.test(message)) return false;
  }
};

// production mode (see build task)
var isProduction = false;
var useSourceMaps = false;

// ignore everything that begins with underscore
var hidden_files = '**/_*.*';
var ignored_files = '!' + hidden_files;

var component_hidden_files = '**/js/**/*.*';
var component_ignored_files = '!' + component_hidden_files;

// VENDOR CONFIG
var vendor = {
  // vendor scripts required to start the app
  base: {source: require('./vendor.base.json'), dest: '../app/assets/js', name: 'base.js'},
  // vendor scripts to make to app work. Usually via lazy loading
  app: {source: require('./vendor.json'), dest: '../app/vendor'}
};

// SOURCES CONFIG
var source = {
  scripts: {
    app: [
      'js/app.preinit.js',
      'js/app.init.js',
      'js/app.config.js',
      'js/app.directive.js',
      'js/app.run.js',
      'js/app.service.js',
      'js/app.provider.js',
      'js/tabs.js',
      'js/sections.js',
      'shared/config/generated-config.js',
      'shared/js/modules/*.js',
      'shared/js/modules/controllers/*.js',
      'shared/js/modules/directives/*.js',
      'shared/js/modules/services/*.js',
      'components/*/js/**/*.js'
    ],
    watch: ['manifest.json', 'js/**/*.js', 'shared/**/*.js', 'components/*/js/**/*.js']
  },
  // templates: {
  //   app: {
  //       files : ['jade/index.jade'],
  //       watch: ['jade/index.jade', hidden_files]
  //   },
  //   views: {
  //       files : ['jade/views/*.jade', 'jade/views/**/*.jade', ignored_files],
  //       watch: ['jade/views/**/*.jade']
  //   },
  //   pages: {
  //       files : ['jade/pages/*.jade'],
  //       watch: ['jade/pages/*.jade']
  //   }
  // },

  styles: {
    app: {
      // , 'components/*/less/*.less'
      source: ['less/app/base.less', 'components/*/less/*.less'],
      dir: ['less/app', 'components'],
      watch: ['less/*.less', 'less/**/*.less', 'components/**/less/*.less', 'components/**/less/**/*.less']

    }
  },

  components: {
    source: [
      'components/**/*.*',
      component_ignored_files,
      '!components/**/config/*.*',
      '!master/shared/js/modules/config.js',
      '!components/*/less/*.*',
      '!components/**/less/**/*.*'
    ],
    dest: 'components',
    watch: [
      'components/**/*.*',
      component_ignored_files,
      '!components/**/config/*.*',
      '!master/shared/js/modules/config.js',
      '!components/**/less/*.*'
    ]
  },

  config: {
    watch: [
      'shared/config/development.json',
      'shared/config/production.json',
      'shared/config/development.json',
      'shared/config/production.json'
    ],
    dest: 'shared/config'
  },

  assets: {source: ['shared/assets/**/*.*'], dest: 'shared/assets', watch: ['shared/assets/**/*.*']}

  //,
  // bootstrap: {
  //   main: 'less/bootstrap/bootstrap.less',
  //   dir:  'less/bootstrap',
  //   watch: ['less/bootstrap/*.less']
  // }
};

// BUILD TARGET CONFIG
var build = {
  scripts: {app: {main: 'app.js', dir: '../app/assets/js'}},
  assets: '../app/shared/assets',
  styles: '../app/assets/css',
  components: {dir: '../app/components'}
};

function stringSrc(filename, string) {
  var src = require('stream').Readable({objectMode: true});
  src._read = function() {
    this.push(new gutil.File({cwd: "", base: "", path: filename, contents: new Buffer(string)}));
    this.push(null);
  };
  return src;
}

//---------------
// TASKS
//---------------

var manifestDirectory = function(manifestFile) {
  return manifestFile.relative.slice(0, '/manifest.json'.length * -1);
};

gulp.task('bundle-manifest', function() {
  var components = [];
  var namespace = [];
  gulp.src('./components/*/manifest.json')
      .pipe(foreach (function(stream, file) {
        var component = manifestDirectory(file);
        components.push(component);
        namespace.push(changeCase.camelCase(component));
        return stream;
      }))
      .pipe(gcallback(function() {
        var tabs = [];
        components.forEach(function(component) {
          tabs.push({component: component, title: changeCase.titleCase(component)});
        });
        stringSrc("tabs.js", 'app.value("tabs", ' + JSON.stringify(tabs) + ');').pipe(gulp.dest("js"));
        var _appNS = 'kubernetesApp.components.';
        var _appSkeleton = require('./js/app.skeleton.json');
        stringSrc("app.preinit.js",
                  _appSkeleton.appSkeleton.replace('%s', '"' + _appNS + namespace.join('", "' + _appNS) + '"'))
            .pipe(gulp.dest("js"));
      }));
});

gulp.task('bundle-manifest-routes', function() {
  var sections = [];
  gulp.src('./components/*/manifest.json')
      .pipe(foreach (function(stream, file) {
        var component = manifestDirectory(file);
        var manifestFile = require(file.path);
        var routes = [];
        if (manifestFile.routes) {
          manifestFile.routes.forEach(function(r) {
            // Hacky deep copy. Modifying manifestFile here will be repeated
            // each time the task is called (consider watch/reload) due to
            // cached file reads.
            var route = JSON.parse(JSON.stringify(r));
            if (route.url) {
              route.url = '/' + component + route.url;
            }
            routes.push(route);
          });
        }
        sections = sections.concat(routes);
        return stream;
      }))
      .pipe(gcallback(function() {
        var output_sections = JSON.stringify(sections);
        var _file_contents = 'app.constant("manifestRoutes", ' + output_sections + ');\n';
        stringSrc("sections.js", _file_contents).pipe(gulp.dest("js"));
      }));
});

// JS APP
gulp.task('scripts:app', ['bundle-manifest', 'bundle-manifest-routes', 'config', 'scripts:app:base']);

// JS APP BUILD
gulp.task('scripts:app:base', function() {
  // Minify and copy all JavaScript (except vendor scripts)
  return gulp.src(source.scripts.app)
      .pipe(useSourceMaps ? sourcemaps.init() : gutil.noop())
      .pipe(concat(build.scripts.app.main))
      .pipe(ngAnnotate())
      .on("error", handleError)
      .pipe(isProduction ? uglify({preserveComments: 'some'}) : gutil.noop())
      .on("error", handleError)
      .pipe(useSourceMaps ? sourcemaps.write() : gutil.noop())
      .pipe(gulp.dest(build.scripts.app.dir));
});

// VENDOR BUILD
gulp.task('scripts:vendor', ['scripts:vendor:base', 'scripts:vendor:app']);

//  This will be included vendor files statically
gulp.task('scripts:vendor:base', function() {

  // Minify and copy all JavaScript (except vendor scripts)
  return gulp.src(vendor.base.source)
      .pipe(expect(vendor.base.source))
      .pipe(uglify())
      .pipe(concat(vendor.base.name))
      .pipe(gulp.dest(vendor.base.dest));
});

// copy file from bower folder into the app vendor folder
gulp.task('scripts:vendor:app', function() {

  var jsFilter = gulpFilter('**/*.js');
  var cssFilter = gulpFilter('**/*.css');

  return gulp.src(vendor.app.source, {base: 'bower_components'})
      .pipe(expect(vendor.app.source))
      .pipe(jsFilter)
      .pipe(uglify())
      .pipe(jsFilter.restore())
      .pipe(cssFilter)
      .pipe(minifyCSS())
      .pipe(cssFilter.restore())
      .pipe(gulp.dest(vendor.app.dest));

});

// APP LESS
gulp.task('styles:app', function() {
  return gulp.src(source.styles.app.source)
      .pipe(foreach (function(stream, file) { return stringSrc('import.less', '@import "' + file.relative + '";\n'); }))
      .pipe(concat('app.less'))
      .pipe(useSourceMaps ? sourcemaps.init() : gutil.noop())
      .pipe(less({paths: source.styles.app.dir}))
      .on("error", handleError)
      .pipe(isProduction ? minifyCSS() : gutil.noop())
      .pipe(useSourceMaps ? sourcemaps.write() : gutil.noop())
      .pipe(gulp.dest(build.styles));
});

// // APP RTL
// gulp.task('styles:app:rtl', function() {
//     return gulp.src(source.styles.app.main)
//         .pipe( useSourceMaps ? sourcemaps.init() : gutil.noop())
//         .pipe(less({
//             paths: [source.styles.app.dir]
//         }))
//         .on("error", handleError)
//         .pipe(flipcss())
//         .pipe( isProduction ? minifyCSS() : gutil.noop() )
//         .pipe( useSourceMaps ? sourcemaps.write() : gutil.noop())
//         .pipe(rename(function(path) {
//             path.basename += "-rtl";
//             return path;
//         }))
//         .pipe(gulp.dest(build.styles));
// });

// Environment based configuration
// https://github.com/kubernetes-ui/kubernetes-ui/issues/21

gulp.task('config', ['config:base', 'config:copy']);

gulp.task('config:base', function() {
  return stringSrc('generated-config.js', 'angular.module("kubernetesApp.config", [])' +
                                              '\n' +
                                              '.constant("ENV", {})').pipe(gulp.dest(source.config.dest));
});

gulp.task('config:copy', function() {
  var environment = argv.env || 'development';  // change this to whatever default environment you need.

  return gulp.src(['shared/config/' + environment + '.json', 'components/**/config/' + environment + '.json'])
      .pipe(jsoncombine('generated-config.js',
                        function(data) {
                          var env = Object.keys(data).reduce(function(result, key) {
                            // Map the key "environment" to "/" and the keys "component/config/environment" to
                            // "component".
                            var newKey = key.replace(environment, '/').replace(/\/config\/\/$/, '');
                            result[newKey] = data[key];
                            return result;
                          }, {});

                          return new Buffer(JSON.stringify({'ENV': env}));
                        }))
      .pipe(ngConstant({name: 'kubernetesApp.config', deps: [], constants: {ngConstant: true}}))
      .pipe(gulp.dest(source.config.dest));
});

gulp.task('copy:components', function() {

  var jsFilter = gulpFilter('**/*.js');
  var cssFilter = gulpFilter('**/*.css');

  del.sync([build.components.dir], {force: true});

  return gulp.src(source.components.source, {base: 'components'})
      .pipe(expect(source.components.source))
      .pipe(jsFilter)
      .pipe(uglify())
      .pipe(jsFilter.restore())
      .pipe(cssFilter)
      .pipe(minifyCSS())
      .pipe(cssFilter.restore())
      .pipe(gulp.dest(build.components.dir));
});

gulp.task('copy:shared-assets', function() {
  del.sync([build.assets], {force: true});

  return gulp.src(source.assets.source, {base: 'shared/assets'})
      .pipe(expect(source.assets.source))
      .pipe(gulp.dest(build.assets));
});

// Assuming there's "version: 1.2.3" in package.json,
// tag the last commit as "v1.2.3"//
gulp.task('tag', function() { return gulp.src(['./package.json']).pipe(tag_version()); });

// // BOOSTRAP
// gulp.task('bootstrap', function() {
//     return gulp.src(source.bootstrap.main)
//         .pipe(less({
//             paths: [source.bootstrap.dir]
//         }))
//         .on("error", handleError)
//         .pipe(gulp.dest(build.styles));
// });

// JADE
// gulp.task('templates:app', function() {
//     return gulp.src(source.templates.app.files)
//         .pipe(changed(build.templates.app, { extension: '.html' }))
//         .pipe(jade())
//         .on("error", handleError)
//         .pipe(prettify({
//             indent_char: ' ',
//             indent_size: 3,
//             unformatted: ['a', 'sub', 'sup', 'b', 'i', 'u']
//         }))
//         // .pipe(htmlify({
//         //     customPrefixes: ['ui-']
//         // }))
//         // .pipe(w3cjs( W3C_OPTIONS ))
//         .pipe(gulp.dest(build.templates.app))
//         ;
// });

// // JADE
// gulp.task('templates:pages', function() {
//     return gulp.src(source.templates.pages.files)
//         .pipe(changed(build.templates.pages, { extension: '.html' }))
//         .pipe(jade())
//         .on("error", handleError)
//         .pipe(prettify({
//             indent_char: ' ',
//             indent_size: 3,
//             unformatted: ['a', 'sub', 'sup', 'b', 'i', 'u']
//         }))
//         // .pipe(htmlify({
//         //     customPrefixes: ['ui-']
//         // }))
//         // .pipe(w3cjs( W3C_OPTIONS ))
//         .pipe(gulp.dest(build.templates.pages))
//         ;
// });

// // JADE
// gulp.task('templates:views', function() {
//     return gulp.src(source.templates.views.files)
//         .pipe(changed(build.templates.views, { extension: '.html' }))
//         .pipe(jade())
//         .on("error", handleError)
//         .pipe(prettify({
//             indent_char: ' ',
//             indent_size: 3,
//             unformatted: ['a', 'sub', 'sup', 'b', 'i', 'u']
//         }))
//         // .pipe(htmlify({
//         //     customPrefixes: ['ui-']
//         // }))
//         // .pipe(w3cjs( W3C_OPTIONS ))
//         .pipe(gulp.dest(build.templates.views))
//         ;
// });

//---------------
// WATCH
//---------------

// Rerun the task when a file changes
gulp.task('watch', function() {
  livereload.listen();

  gulp.watch(source.scripts.watch, ['scripts:app']);
  gulp.watch(source.styles.app.watch, ['styles:app']);
  gulp.watch(source.components.watch, ['copy:components']);
  gulp.watch(source.assets.watch, ['copy:shared-assets']);
  // gulp.watch(source.templates.pages.watch,   ['templates:pages']);
  // gulp.watch(source.templates.views.watch,   ['templates:views']);
  // gulp.watch(source.templates.app.watch,     ['templates:app']);

  gulp.watch([

                 '../app/**'

  ])
      .on('change', function(event) {

        livereload.changed(event.path);

      });

});

//---------------
// DEFAULT TASK
//---------------

// build for production (minify)
gulp.task('build', ['prod', 'default']);
gulp.task('prod', function() { isProduction = true; });

// build with sourcemaps (no minify)
gulp.task('sourcemaps', ['usesources', 'default']);
gulp.task('usesources', function() { useSourceMaps = true; });

// default (no minify)
gulp.task('default', gulpsync.sync(['scripts:vendor', 'copy:components', 'scripts:app', 'start']), function() {

  gutil.log(gutil.colors.cyan('************'));
  gutil.log(gutil.colors.cyan('* All Done *'),
            'You can start editing your code, LiveReload will update your browser after any change..');
  gutil.log(gutil.colors.cyan('************'));

});

gulp.task('start', [
  'styles:app',
  'copy:components',
  'copy:shared-assets',
  // 'templates:app',
  // 'templates:pages',
  // 'templates:views',
  'watch'
]);

gulp.task('done', function() {
  console.log('All Done!! You can start editing your code, LiveReload will update your browser after any change..');
});

// Error handler
function handleError(err) {
  console.log(err.toString());
  this.emit('end');
}

// // Mini gulp plugin to flip css (rtl)
// function flipcss(opt) {

//   if (!opt) opt = {};

//   // creating a stream through which each file will pass
//   var stream = through.obj(function(file, enc, cb) {
//     if(file.isNull()) return cb(null, file);

//     if(file.isStream()) {
//         console.log("todo: isStream!");
//     }

//     var flippedCss = flip(String(file.contents), opt);
//     file.contents = new Buffer(flippedCss);
//     cb(null, file);
//   });

//   // returning the file stream
//   return stream;
// }
