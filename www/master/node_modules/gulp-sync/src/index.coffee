'use strict'
groupCount = 0
module.exports = (gulp)->
    async = (tasks, prefix = 'sync group')->
        prefix += groupCount++ if prefix == 'sync group'
        result = []
        syncCount = 0
        for task in tasks
            if Array.isArray task
                task = sync(task, prefix + ':' + syncCount)[0]
                syncCount++
            result.push task

        result

    sync = (tasks, prefix = 'sync group')->
        prefix += groupCount++ if prefix == 'sync group'
        deps = []
        for task, index in tasks
            taskName = prefix + ':' + index
            if Array.isArray task
                task = async task, taskName
            else
                task = [task]

            do (taskName, deps, task)->
                gulp.task taskName, deps, (cb)->
                    check = task.concat()
                    gulp.addListener 'task_stop', onStop = (e)->
                        if -1 != i = check.indexOf e.task
                            check.splice i, 1
                            if check.length == 0
                                gulp.removeListener 'task_stop', onStop
                                cb()
                    gulp.start.apply gulp, task
            deps = [taskName]

        deps

    {async, sync}

