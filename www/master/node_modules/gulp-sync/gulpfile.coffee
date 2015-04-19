'use strict'

gulp = require 'gulp'
coffee = require 'gulp-coffee'
mocha = require 'gulp-mocha'

gulp.task 'build', ->
    gulp.src 'src/index.coffee'
        .pipe coffee bare: true
        .pipe gulp.dest 'lib'

gulp.task 'test', ['build'], ->
    gulp.src 'test/index.coffee', read: false
        .pipe mocha()

gulp.task 'default', ['build']
