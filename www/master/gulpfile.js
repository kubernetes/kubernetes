var gulp = require('gulp'), concat = require('gulp-concat'), uglify = require('gulp-uglify'),
    less = require('gulp-less'), path = require('path'),
    livereload = require('gulp-livereload'),
    path = require('path'), changed = require('gulp-changed'), prettify = require('gulp-html-prettify'),
    w3cjs = require('gulp-w3cjs'), rename = require('gulp-rename'),
    through = require('through2'), gutil = require('gulp-util'), htmlify = require('gulp-angular-htmlify'),
    minifyCSS = require('gulp-minify-css'), gulpFilter = require('gulp-filter'), expect = require('gulp-expect-file'),
    gulpsync = require('gulp-sync')(gulp), ngAnnotate = require('gulp-ng-annotate'),
    sourcemaps = require('gulp-sourcemaps'), del = require('del'), jsoncombine = require('gulp-jsoncombine'),
    ngConstant = require('gulp-ng-constant'), foreach = require('gulp-foreach'),
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
    if (/Element head is missing a required instance of child element title/.test(message)) { return false; }
    if (/Attribute .+ not allowed on element .+ at this point/.test(message)) { return false; }
    if (/Element .+ not allowed as child of element .+ in this context/.test(message)) { return false; }
    if (/Comments seen before doctype./.test(message)) { return false; }
  }
};

// production mode (see build task)
var isProduction = false;
var useSourceMaps = false;

// ignore everything that begins with underscore
var hidden_files = '**/_*.*';
var ignored_files = '!' + hidden_files;

var output_folder = '../app';

// VENDOR CONFIG
var vendor = {
  // vendor scripts required to start the app
  base: {
    source: require('./vendor.base.json'), 
    dest: '../app/assets/js', 
    name: 'base.js'
  },
  // vendor scripts to make to app work. Usually via lazy loading
  app: {
    // instead of the bower downloaded versions of some files, we
    // pull hand edited versions from the shared/vendor directory.
    source: require('./vendor.json'),
    dest: '../app/vendor'
  }
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
      'components/**/js/**/*.js'
    ],
    dest: {
      name: 'app.js', 
      dir: '../app/assets/js'
    },
    watch: [
      'manifest.json', 
      'js/**/*.js', 
      'shared/**/*.js', 
      'shared/config/*.json',
      'components/*/js/**/*.js',
      'components/*/config/*.json'
      ]
  },

  styles: {
    app: {
      source: ['less/app/base.less', 'components/*/less/*.less'],
      paths: ['less/app', 'components'],
      dest: '../app/assets/css',
      watch: ['less/*.less', 'less/**/*.less', 'components/**/less/*.less', 'components/**/less/**/*.less']

    }
  },

  html: {
    app: {
      source: ['shared/index.html'],
      dest: '../app'
    },
    views: {
      source: ['shared/views/**/*.*'],
      dest: '../app'
    },
    watch: ['shared/index.html', 'shared/views/**/*.*']
  },

  components: {
    source: [
      'components/**/*.*',
      '!components/**/js/**/*.*',
      '!components/**/config/**/*.*',
      '!components/**/protractor/**/*.*',
      '!components/**/test/**/*.*',
      '!components/**/less/**/*.*',
      '!components/**/README.md'
    ],
    dest: '../app/components',
    watch: [
      'components/**/*.*',
      '!components/**/js/**/*.*',
      '!components/**/config/**/*.*',
      '!components/**/protractor/**/*.*',
      '!components/**/test/**/*.*',
      '!components/**/less/**/*.*',
      '!components/**/README.md'
    ]
  },

  config: {
    dest: 'shared/config'
  },

  assets: {
    source: ['shared/assets/**/*.*'], 
    dest: '../app/assets', 
    watch: ['shared/assets/**/*.*']
  }
};

function stringSrc(filename, string) {
  var src = require('stream').Readable({objectMode: true});
  src._read = function() {
    this.push(new gutil.File({cwd: "", base: "", path: filename, contents: new Buffer(string)}));
    this.push(null);
  };
  return src;
}

// Error handler
function handleError(err) {
  console.log(err.toString());
  this.emit('end');
  process.exit(1);
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
gulp.task('scripts:app', gulpsync.sync(['bundle-manifest', 'bundle-manifest-routes', 'config', 'scripts:app:base']));

// JS APP BUILD
gulp.task('scripts:app:base', function() {
  // Minify and copy all JavaScript (except vendor scripts)
  return gulp.src(source.scripts.app)
      .pipe(useSourceMaps ? sourcemaps.init() : gutil.noop())
      .pipe(concat(source.scripts.dest.name))
      .pipe(ngAnnotate())
      // Now that we run a production build, uglification is breaking angular injection, 
      // so disable it for now.
      // TODO: Find out which dependencies are not string based and upgrade them accordingly.
      // .pipe(isProduction ? uglify({preserveComments: 'some'}) : gutil.noop())
      .pipe(useSourceMaps ? sourcemaps.write() : gutil.noop())
      .pipe(gulp.dest(source.scripts.dest.dir))
      .on("error", handleError);
});

// VENDOR BUILD
gulp.task('scripts:vendor', gulpsync.sync(['scripts:vendor:base', 'scripts:vendor:app']));

//  This will be included vendor files statically
gulp.task('scripts:vendor:base', function() {

  // Minify and copy all JavaScript (except vendor scripts)
  return gulp.src(vendor.base.source)
      .pipe(expect({ errorOnFailure: true }, vendor.base.source))
      .pipe(isProduction ? uglify() : gutil.noop())
      .pipe(concat(vendor.base.name))
      .pipe(gulp.dest(vendor.base.dest))
      .on("error", handleError);
});

// copy file from bower folder into the app vendor folder
gulp.task('scripts:vendor:app', function() {

  var jsFilter = gulpFilter('**/*.js');
  var cssFilter = gulpFilter('**/*.css');

  return gulp.src(vendor.app.source)
      .pipe(expect({ errorOnFailure: true }, vendor.app.source))
      .pipe(jsFilter)
      .pipe(isProduction ? uglify() : gutil.noop())
      .pipe(jsFilter.restore())
      .pipe(cssFilter)
      .pipe(isProduction ? minifyCSS() : gutil.noop())
      .pipe(cssFilter.restore())
      .pipe(gulp.dest(vendor.app.dest))
      .on("error", handleError);

});

// APP LESS
gulp.task('styles:app', function() {
  return gulp.src(source.styles.app.source)
      .pipe(foreach (function(stream, file) { return stringSrc('import.less', '@import "' + file.relative + '";\n'); }))
      .pipe(concat('app.less'))
      .pipe(useSourceMaps ? sourcemaps.init() : gutil.noop())
      .pipe(less({paths: source.styles.app.paths}))
      .pipe(isProduction ? minifyCSS() : gutil.noop())
      .pipe(useSourceMaps ? sourcemaps.write() : gutil.noop())
      .pipe(gulp.dest(source.styles.app.dest))
      .on("error", handleError);
});

gulp.task('config', gulpsync.sync(['config:base', 'config:copy']));

gulp.task('config:base', function() {
  return stringSrc('generated-config.js', 'angular.module("kubernetesApp.config", [])' +
                                              '\n' +
                                              '.constant("ENV", {})').pipe(gulp.dest(source.config.dest));
});

gulp.task('config:copy', function() {
  var environment = isProduction ? 'production' : 'development';
  return gulp.src(['shared/config/' + environment + '.json', 'components/**/config/' + environment + '.json'])
    .pipe(expect({ errorOnFailure: true }, 'shared/config/' + environment + '.json'))
    .on("error", handleError)
    .pipe(jsoncombine('generated-config.js',
      function(data) {
        var env = Object.keys(data).reduce(function(result, key) {
          // Map the key "environment" to "/" and the keys "component/config/environment" to "component".
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

  return gulp.src(source.components.source, {base: 'components'})
      .pipe(expect({ errorOnFailure: true }, source.components.source))
      .pipe(jsFilter)
      .pipe(isProduction ? uglify() : gutil.noop())
      .pipe(jsFilter.restore())
      .pipe(cssFilter)
      .pipe(isProduction ? minifyCSS() : gutil.noop())
      .pipe(cssFilter.restore())
      .pipe(gulp.dest(source.components.dest))
      .on("error", handleError);
});

gulp.task('copy:shared-assets', function() {
  return gulp.src(source.assets.source, {base: 'shared/assets'})
      .pipe(gulp.dest(source.assets.dest));
});

// Assuming there's "version: 1.2.3" in package.json,
// tag the last commit as "v1.2.3"//
gulp.task('tag', function() { return gulp.src(['./package.json']).pipe(tag_version()); });

// VIEWS
gulp.task('content:html', gulpsync.sync(['content:html:app', 'content:html:views']));

gulp.task('content:html:app', function() {
  return gulp.src(source.html.app.source, {base: 'shared'})
    .pipe(prettify({
        indent_char: ' ',
        indent_size: 4,
        unformatted: ['a', 'sub', 'sup', 'b', 'i', 'u']
    }))
    .pipe(gulp.dest(source.html.app.dest))
    .on("error", handleError);
});

gulp.task('content:html:views', function() {
  return gulp.src(source.html.views.source, {base: 'shared'})
    .pipe(prettify({
        indent_char: ' ',
        indent_size: 4,
        unformatted: ['a', 'sub', 'sup', 'b', 'i', 'u']
    }))
    .pipe(gulp.dest(source.html.views.dest))
    .on("error", handleError);
});

//---------------
// WATCH
//---------------

// Rerun the task when a file changes
gulp.task('watch', function() {
  livereload.listen();

  gulp.watch(source.html.watch, ['content:html']);
  gulp.watch(source.scripts.watch, ['scripts:app']);
  gulp.watch(source.styles.app.watch, ['styles:app']);
  gulp.watch(source.components.watch, ['copy:components']);
  gulp.watch(source.assets.watch, ['copy:shared-assets']);

  gulp.watch(['../app/**'])
      .on('change', function(event) {
        livereload.changed(event.path);
      });
});

//---------------
// ENTRY POINTS
//---------------

// build for production (minify)
gulp.task('build', gulpsync.sync(['prod', 'clean', 'compile']));
gulp.task('prod', function() { isProduction = true; });

// build with sourcemaps (no minify)
gulp.task('sourcemaps', gulpsync.sync(['usesources', 'compile']));
gulp.task('usesources', function() { useSourceMaps = true; });

// build for development (no minify)
gulp.task('default', gulpsync.sync(['clean', 'compile', 'watch']), function() {
  gutil.log(gutil.colors.cyan('************'));
  gutil.log('You can start editing your code. LiveReload will update your browser after any change.');
  gutil.log(gutil.colors.cyan('************'));
});

gulp.task('clean', function() {
  del.sync(['shared/config/generated-config.js'], {force: true});
  del.sync([output_folder], {force: true});
});

gulp.task('compile', gulpsync.sync([
  'copy:shared-assets',
  'copy:components',
  'content:html',
  'scripts:vendor', 
  'scripts:app', 
  'styles:app'
]));
