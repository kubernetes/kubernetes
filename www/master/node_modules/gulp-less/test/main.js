var should = require('should');
var less = require('../');
var gutil = require('gulp-util');
var fs = require('fs');
var pj = require('path').join;

function createVinyl(lessFileName, contents) {
  var base = pj(__dirname, 'fixtures');
  var filePath = pj(base, lessFileName);

  return new gutil.File({
    cwd: __dirname,
    base: base,
    path: filePath,
    contents: contents || fs.readFileSync(filePath)
  });
}

describe('gulp-less', function () {
  describe('less()', function () {
    it('should pass file when it isNull()', function (done) {
      var stream = less();
      var emptyFile = {
        isNull: function () { return true; }
      };
      stream.on('data', function (data) {
        data.should.equal(emptyFile);
        done();
      });
      stream.write(emptyFile);
    });

    it('should emit error when file isStream()', function (done) {
      var stream = less();
      var streamFile = {
        isNull: function () { return false; },
        isStream: function () { return true; }
      };
      stream.on('error', function (err) {
        err.message.should.equal('Streaming not supported');
        done();
      });
      stream.write(streamFile);
    });

    it('should compile single less file', function (done) {
      var lessFile = createVinyl('buttons.less');

      var stream = less();
      stream.on('data', function (cssFile) {
        should.exist(cssFile);
        should.exist(cssFile.path);
        should.exist(cssFile.relative);
        should.exist(cssFile.contents);
        cssFile.path.should.equal(pj(__dirname, 'fixtures', 'buttons.css'));
        String(cssFile.contents).should.equal(
          fs.readFileSync(pj(__dirname, 'expect/buttons.css'), 'utf8'));
        done();
      });
      stream.write(lessFile);
    });

    it('should emit error when less contains errors', function (done) {
      var stream = less();
      var errorFile = createVinyl('somefile.less',
        new Buffer('html { color: @undefined-variable; }'));
      stream.on('error', function (err) {
        err.message.should.equal('variable @undefined-variable is undefined in file '+errorFile.path+' line no. 1');
        done();
      });
      stream.write(errorFile);
    });

    it('should continue to process next files when less error occurs', function (done) {
      var stream = less();

      var errorFile = createVinyl('somefile.less',
        new Buffer('html { color: @undefined-variable; }'));
      var normalFile = createVinyl('buttons.less');

      var errorHandled = false;
      var dataHandled = false;

      stream.on('error', function (err) {
        err.message.should.equal('variable @undefined-variable is undefined in file '+errorFile.path+' line no. 1');
        errorHandled = true;
        if (dataHandled) {
          done();
        }
      });
      stream.on('data', function (cssFile) {
        dataHandled = true;
        if (errorHandled) {
          done();
        }
      });
      stream.write(errorFile);
      stream.write(normalFile);
    });

    it('should compile multiple less files', function (done) {
      var files = [
        createVinyl('buttons.less'),
        createVinyl('forms.less'),
        createVinyl('normalize.less')
      ];

      var stream = less();
      var count = files.length;
      stream.on('data', function (cssFile) {
        should.exist(cssFile);
        should.exist(cssFile.path);
        should.exist(cssFile.relative);
        should.exist(cssFile.contents);
        if (!--count) { done(); }
      });

      files.forEach(function (file) {
        stream.write(file);
      });
    });

    it('should provide target filename to sourcemap', function (done) {
      var files = [
        createVinyl('buttons.less'),
        createVinyl('forms.less'),
        createVinyl('normalize.less')
      ] .map(function (file) {
        file.sourceMap = {
          file: '',
          version : 3,
          sourceRoot : '',
          sources: [],
          names: [],
          mappings: ''
        };

        return file;
      });

      var stream = less();
      var count = files.length;
      stream.on('data', function (cssFile) {
        should.exist(cssFile.sourceMap.file);
      });
      stream.on('end', done);

      files.forEach(function (file) {
        stream.write(file);
      });
      stream.end();
    });
  });
});
