'use strict'

should = require 'should'

describe 'gulp-sync', ->
    gulp = null
    gulpsync = null
    tasks1 = ['a', 'b', 'c']
    tasks2 = ['a', ['b-1', 'b-2'], ['c-1', 'c-2', ['c-3-1', 'c-3-2']]]
    deepCheck = (tasks)->
        for task in tasks
            # console.log task
            deps = gulp.tasks[task]?.dep
            # console.log deps
            if deps
                for dep in deps
                    # console.log dep
                    should.exist gulp.tasks[dep]
                    deepCheck deps
    createTasks = (tasks, isAsync)->
        wait = 300
        for task in tasks
            if Array.isArray task
                createTasks task, !isAsync
            else
                do (task)->
                    gulp.task task, (cb)->
                        setTimeout ->
                            gulp.tasks[task].running.should.equal true
                            console.log task, if isAsync then 'async' else 'sync'
                            cb()
                        , wait -= 100

    beforeEach ->
        delete require.cache[require.resolve 'gulp']
        delete require.cache[require.resolve '../lib/']
        gulp = require 'gulp'
        gulpsync = require('../lib/') gulp


    it 'async flat', (done)->
        console.log 'async flat'
        createTasks tasks1, true
        t = gulpsync.async tasks1
        should.exist t
        Array.isArray(t).should.equal true
        t[0].should.equal tasks1[0]
        t[1].should.equal tasks1[1]
        t[2].should.equal tasks1[2]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'async flat without custom group name', (done)->
        console.log 'async flat without custom group name'
        createTasks tasks1, true
        t = gulpsync.async tasks1
        should.exist t
        Array.isArray(t).should.equal true
        t[0].should.equal tasks1[0]
        t[1].should.equal tasks1[1]
        t[2].should.equal tasks1[2]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'async flat with custom group name', (done)->
        console.log 'async flat with custom group name'
        createTasks tasks1, true
        t = gulpsync.async tasks1, 'custom group name'
        should.exist t
        Array.isArray(t).should.equal true
        t[0].should.equal tasks1[0]
        t[1].should.equal tasks1[1]
        t[2].should.equal tasks1[2]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'async deep', (done)->
        console.log 'async deep'
        createTasks tasks2, true
        t = gulpsync.async tasks2
        should.exist t
        Array.isArray(t).should.equal true
        t[0].should.equal tasks2[0]
        should.exist gulp.tasks[t[1]]
        should.exist gulp.tasks[t[2]]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'async deep without custom group name', (done)->
        console.log 'async deep without custom group name'
        createTasks tasks2, true
        t = gulpsync.async tasks2
        should.exist t
        Array.isArray(t).should.equal true
        t[0].should.equal tasks2[0]
        should.exist gulp.tasks[t[1]]
        should.exist gulp.tasks[t[2]]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks


    it 'async deep with custom group name', (done)->
        console.log 'async deep with custom group name'
        createTasks tasks2, true
        t = gulpsync.async tasks2, 'custom group name'
        should.exist t
        Array.isArray(t).should.equal true
        t[0].should.equal tasks2[0]
        should.exist gulp.tasks[t[1]]
        should.exist gulp.tasks[t[2]]
        t[1].split(':')[0].should.equal 'custom group name'
        t[2].split(':')[0].should.equal 'custom group name'
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks


    it 'sync flat', (done)->
        console.log 'sync flat'
        createTasks tasks1, false
        t = gulpsync.sync tasks1
        should.exist t
        Array.isArray(t).should.equal true
        t.length.should.equal 1
        should.exist gulp.tasks[t[0]]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'sync flat without custom group name', (done)->
        console.log 'sync flat without custom group name'
        createTasks tasks1, false
        t = gulpsync.sync tasks1
        should.exist t
        Array.isArray(t).should.equal true
        t.length.should.equal 1
        should.exist gulp.tasks[t[0]]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'sync flat with custom group name', (done)->
        console.log 'sync flat with custom group name'
        createTasks tasks1, false
        t = gulpsync.sync tasks1, 'custom group name'
        should.exist t
        Array.isArray(t).should.equal true
        t.length.should.equal 1
        should.exist gulp.tasks[t[0]]
        t[0].split(':')[0].should.equal 'custom group name'
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'sync deep', (done)->
        console.log 'sync deep'
        createTasks tasks2, false
        t = gulpsync.sync tasks2
        should.exist t
        Array.isArray(t).should.equal true
        t.length.should.equal 1
        should.exist gulp.tasks[t[0]]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'sync deep without custom group name', (done)->
        console.log 'sync deep without custom group name'
        createTasks tasks2, false
        t = gulpsync.sync tasks2
        should.exist t
        Array.isArray(t).should.equal true
        t.length.should.equal 1
        should.exist gulp.tasks[t[0]]
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'sync deep with custom group name', (done)->
        console.log 'sync deep with custom group name'
        createTasks tasks2, false
        t = gulpsync.sync tasks2, 'custom group name'
        should.exist t
        Array.isArray(t).should.equal true
        t.length.should.equal 1
        should.exist gulp.tasks[t[0]]
        t[0].split(':')[0].should.equal 'custom group name'
        deepCheck t
        gulp.task 'test', t, ->
            done()
        gulp.start 'test'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'mix and multiple without custom group name', (done)->
        console.log 'mix and multiple without custom group name'
        createTasks tasks1, true
        createTasks tasks2, false
        t1 = gulpsync.sync tasks1
        t2 = gulpsync.sync tasks2
        should.exist t1
        should.exist t2
        Array.isArray(t1).should.equal true
        Array.isArray(t2).should.equal true
        t1.length.should.equal 1
        t2.length.should.equal 1
        should.exist gulp.tasks[t1[0]]
        should.exist gulp.tasks[t2[0]]
        deepCheck t1
        deepCheck t2
        gulp.task 'test1', t1, ->
            gulp.start 'test2'
        gulp.task 'test2', t2, ->
            done()
        gulp.start 'test1'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

    it 'mix and multiple with custom group name', (done)->
        console.log 'mix and multiple with custom group name'
        createTasks tasks1, true
        createTasks tasks2, false
        t1 = gulpsync.sync tasks1, 'custom group name1'
        t2 = gulpsync.sync tasks2, 'custom group name2'
        should.exist t1
        should.exist t2
        Array.isArray(t1).should.equal true
        Array.isArray(t2).should.equal true
        t1.length.should.equal 1
        t2.length.should.equal 1
        should.exist gulp.tasks[t1[0]]
        should.exist gulp.tasks[t2[0]]
        t1[0].split(':')[0].should.equal 'custom group name1'
        t2[0].split(':')[0].should.equal 'custom group name2'
        deepCheck t1
        deepCheck t2
        gulp.task 'test1', t1, ->
            gulp.start 'test2'
        gulp.task 'test2', t2, ->
            done()
        gulp.start 'test1'
        # console.log '---------------'
        # console.log t
        # console.log gulp.tasks

