/*jshint node:true */
/*global describe:false, it:false */
"use strict";

var consume = require('../');
var Stream = require('stream');
var Readable = Stream.Readable;
var Writable = Stream.Writable;
var Duplex = Stream.Duplex;
var should = require('should');
var through = require('through2');
require('mocha');

describe('stream-consume', function() {

    it('should cause a Readable stream to complete if it\'s not piped anywhere', function(done) {
        var rs = new Readable({highWaterMark: 2});
        var a = 0;
        var ended = false;
        rs._read = function() {
            if (a++ < 100) {
                rs.push(a + "");
            } else {
                ended = true;
                rs.push(null);
            }
        };

        rs.on("end", function() {
            a.should.be.above(99);
            ended.should.be.true;
            done();
        });

        consume(rs);
    });

    it('should work with Readable streams in objectMode', function(done) {
        var rs = new Readable({highWaterMark: 2, objectMode: true});
        var a = 0;
        var ended = false;
        rs._read = function() {
            if (a++ < 100) {
                rs.push(a);
            } else {
                ended = true;
                rs.push(null);
            }
        };

        rs.on("end", function() {
            a.should.be.above(99);
            ended.should.be.true;
            done();
        });

        consume(rs);
    });

    it('should not interfere with a Readable stream that is piped somewhere', function(done) {
        var rs = new Readable({highWaterMark: 2});
        var a = 0;
        var ended = false;
        rs._read = function() {
            if (a++ < 100) {
                rs.push(".");
            } else {
                ended = true;
                rs.push(null);
            }
        };

        var sizeRead = 0;
        var ws = new Writable({highWaterMark: 2});
        ws._write = function(chunk, enc, next) {
            sizeRead += chunk.length;
            next();
        }

        ws.on("finish", function() {
            a.should.be.above(99);
            ended.should.be.true;
            sizeRead.should.equal(100);
            done();
        });

        rs.pipe(ws);

        consume(rs);
    });

    it('should not interfere with a Writable stream', function(done) {
        var rs = new Readable({highWaterMark: 2});
        var a = 0;
        var ended = false;
        rs._read = function() {
            if (a++ < 100) {
                rs.push(".");
            } else {
                ended = true;
                rs.push(null);
            }
        };

        var sizeRead = 0;
        var ws = new Writable({highWaterMark: 2});
        ws._write = function(chunk, enc, next) {
            sizeRead += chunk.length;
            next();
        }

        ws.on("finish", function() {
            a.should.be.above(99);
            ended.should.be.true;
            sizeRead.should.equal(100);
            done();
        });

        rs.pipe(ws);

        consume(ws);
    });

    it('should handle a Transform stream', function(done) {
        var rs = new Readable({highWaterMark: 2});
        var a = 0;
        var ended = false;
        rs._read = function() {
            if (a++ < 100) {
                rs.push(".");
            } else {
                ended = true;
                rs.push(null);
            }
        };

        var sizeRead = 0;
        var flushed = false;
        var ts = through({highWaterMark: 2}, function(chunk, enc, cb) {
            sizeRead += chunk.length;
            this.push(chunk);
            cb();
        }, function(cb) {
            flushed = true;
            cb();
        });

        ts.on("end", function() {
            a.should.be.above(99);
            ended.should.be.true;
            sizeRead.should.equal(100);
            flushed.should.be.true;
            done();
        });

        rs.pipe(ts);

        consume(ts);
    });

    it('should handle a classic stream', function(done) {
        var rs = new Stream();
        var ended = false;
        var i;

        rs.on("end", function() {
            ended.should.be.true;
            done();
        });

        consume(rs);

        for (i = 0; i < 100; i++) {
            rs.emit("data", i);
        }
        ended = true;
        rs.emit("end");
    });

});
