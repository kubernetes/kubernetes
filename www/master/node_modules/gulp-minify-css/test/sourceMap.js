var gulp = require('gulp'),
  expect = require('chai').expect,
  minifyCSS = require('../'),
  CleanCSS = require('clean-css'),
  sourceMaps = require('gulp-sourcemaps'),
  applySourceMap = require('vinyl-sourcemaps-apply'),
  gutil = require('gulp-util'),
  es = require('event-stream'),
  Stream = require('stream'),
  path = require('path'),
  fs = require('fs');

require('mocha');

describe('gulp-minify-css source map', function() {
  var opts = {
    keepSpecialComments: 1,
    keepBreaks: true
  };
  
  describe('with buffers and gulp-sourcemaps', function() {
    var filename = path.join(__dirname, './fixture/sourcemap.css');
    
    it('should generate source map with correct mapping', function(done) {
      var write = sourceMaps.write();
      
      write.on('data', function (file) {
        expect(';AAAA,WACE,kBCKA,YACA;ACPF,WACE,wBACA,kBACA,gBACA').to.be.equal(file.sourceMap.mappings);
        var contents = file.contents.toString();
        expect(/sourceMappingURL=data:application\/json;base64/.test(contents)).to.be.equal(true);
        done();
      });
      
      gulp.src(filename)
      .pipe(sourceMaps.init())
      .pipe(minifyCSS(opts))
      .pipe(write);
    });
    
    it('should generate source map with correct file property', function(done) {      
      var write = sourceMaps.write();
      
      write.on('data', function (file) {
        expect(file.sourceMap).to.have.property('file');
        expect(file.sourceMap.file).to.be.equal('sourcemap.css');
        done();
      });
      
      gulp.src(filename)
      .pipe(sourceMaps.init())
      .pipe(minifyCSS(opts))
      .pipe(write);
    });
    
    it('should generate source map with correct sources', function(done) {      
      var write = sourceMaps.write();
      
      write.on('data', function (file) {
        expect(file.sourceMap).to.have.property('sources').with.length(3);
        done();
      });
      
      gulp.src(filename)
      .pipe(sourceMaps.init())
      .pipe(minifyCSS(opts))
      .pipe(write);
    });
  });
});
