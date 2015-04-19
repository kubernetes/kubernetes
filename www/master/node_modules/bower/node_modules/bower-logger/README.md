# bower-logger [![Build Status](https://secure.travis-ci.org/bower/logger.png?branch=master)](http://travis-ci.org/bower/logger)

The logger used in the various architecture components of Bower.


## Usage

### .error(id, message, data)

Alias to `.log('error', id. message, data)`


### .conflict(id, message, data)

Alias to `.log('conflict', id. message, data)`


### .warn(id, message, data)

Alias to `.log('warn', id. message, data)`


### .action(id, message, data)

Alias to `.log('action', id. message, data)`


### .info(id, message, data)

Alias to `.log('info', id. message, data)`


### .debug(id, message, data)

Alias to `.log('debug', id. message, data)`


### .log(level, id, message, data)

Emits a `log` event, with an object like so:

```js
logger.log('warn', 'foo', 'bar', { dog: 'loves cat' })
{
    level: 'warn',
    id: 'foo',
    message: 'bar',
    data: {
        dog: 'loves cat'
    }
}
```


### .prompt(prompts, callback)

Emits a `prompt` event with an array of `prompts` with a `callback`.   
`prompts` can be an object or an array of objects. The `callback` will be called with an
the answer or an object of answers (if prompts was only one or an array respectively).
The `callback` is guaranteed to run only once.

```js
logger.on('prompt', function (prompts, callback) {
    // "prompts" is always an array of prompts
    // Call "callback" with an object of answers when done

    // In this example, we will use the inquirer module to do the
    // prompting for us
    inquirer(prompts, callback);
})

logger.prompt({
    type: 'input'  // Can be 'input', 'confirm' or 'password'
    message: 'Type something',
    validate: function (value) {
        if (value !== 'I am awesome') {
            return 'You must type "I am awesome"'
        }

        return true;
    }
}, function (err, answer) {
    // Error will only happen on unsupported 'type'
    if (err) {
        return console.error(err.message);
    }

    console.log(answer);
});


```


### .pipe(logger)

Pipes all logger events to another logger.   
Basically all events emitted with `.emit()` will get piped.


### .geminate()

Creates a new logger that pipes events to the parent logger.   
Alias for `(new Logger()).pipe(logger)`.


### .intercept(fn)

Intercepts `log` events, calling `fn` before listeners of the instance.


### #LEVELS

A static property that contains an object where keys are the recognized log levels and values their importance.   
The higher the importance, the more important the level is.


## License

Released under the [MIT License](http://www.opensource.org/licenses/mit-license.php).
