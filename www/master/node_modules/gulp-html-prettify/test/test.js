var gulp = require('gulp');
var expect = require('chai').expect;
var task = require('../');
var html = require('html');
var es = require('event-stream');
var path = require('path');

require('mocha');

describe('gulp-html-prettify compilation', function(){

  'use strict';

  describe('prettify', function(){

    var filename = path.join(__dirname, './fixtures/test.html');
    var opts = {indent_char: ' ', indent_size: 2};

    function expectStream(done, options){
      options = options || {};
      return es.map(function(file){
        var expected = html.prettyPrint('<div><p>String</p></div>', options);        
        expect(expected).to.equal(String(file.contents));
        done();
      });
    }

    it('should prettify HTML', function(done){
      gulp.src(filename)
        .pipe(task(opts))
        .pipe(expectStream(done, opts));
    });

  });

});