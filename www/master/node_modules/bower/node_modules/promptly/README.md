# promptly

[![Build Status](https://secure.travis-ci.org/IndigoUnited/node-promptly.png)](http://travis-ci.org/IndigoUnited/node-promptly.png)

Simple command line prompting utility.

## Installation

`$ npm install promptly`


## API


Note that the `options` argument is optional for all the commands.


### .prompt(message, opts, fn)

Prompts for a value, printing the `message` and waiting for the input.   
When done, calls `fn` with `error` and `value`.

Default options:
```js
{
    // The default value. If not supplied, the input is mandatory
    'default': null,
    // Automatically trim the input
    'trim': true,
    // A validator or an array of validators.
    'validator': null,
    // Automatically retry if a validator fails
    'retry': true,
    // Do not print what the user types
    'silent': false,
    // Input and output streams to read and write to
    'input': process.stdin,
    'output': process.stdout
}
```

The validators have two purposes:
```js
function (value) {
    // Validation example, throwing an error when invalid
    if (value.length !== 2) {
        throw new Error('Length must be 2');
    }

    // Parse the value, modifying it
    return value.replace('aa', 'bb');
}
```

Example usages

Ask for a name:
```js
promptly.prompt('Name: ', function (err, value) {
    // err is always null in this case, because no validators are set
    console.log(value);
});
```

Ask for a name with a constraint (non-empty value and length > 2):

```js
var validator = function (value) {
    if (value.length < 2) {
        throw new Error('Min length of 2');
    }

    return value;
};

promptly.prompt('Name: ', { validator: validator }, function (err, value) {
    // Since retry is true by default, err is always null
    // because promptly will be prompting for a name until it validates
    // Between each prompt, the error message from the validator will be printed
    console.log('Name is:', value);
});
```

Same as above but do not retry automatically:

```js
var validator = function (value) {
    if (value.length < 2) {
        throw new Error('Min length of 2');
    }

    return value;
};

promptly.prompt('Name: ', { validator: validator, retry: false }, function (err, value) {
    if (err) {
        console.error('Invalid name:', e.message);
        // Manually call retry
        // The passed error has a retry method to easily prompt again.
        return err.retry();
    }

    console.log('Name is:', value);
});
```

### .confirm(message, opts, fn)

Ask the user to confirm something.   
Calls `fn` with `error` and `value` (true or false).

Truthy values are: `y`, `yes` and `1`.   
Falsy values are `n`, `no`, and `0`.   
Comparison is made in a case insensitive way.

Example usage:

```js
promptly.confirm('Are you sure? ', function (err, value) {
    console.log('Answer:', value);
});
```


### .choose(message, choices, opts, fn)

Ask the user to choose between multiple `choices` (array of choices).   
Calls `fn` with `error` and `value`.

Example usage:

```js
promptly.choose('Do you want an apple or an orange? ', ['apple', 'orange'], function (err, value) {
    console.log('Answer:', value);
});
```


### .password(message, opts, fn)

Prompts for a password, printing the `message` and waiting for the input.   
When available, calls `fn` with `error` and `value`.

The available options are the same, except that `trim` and `silent` default to `false` and `default` is an empty string (to allow empty passwords).

Example usage:

```js
promptly.password('Type a password: ', function (err, value) {
    console.log('Password is:', value);
});
```


## License

Released under the [MIT License](http://www.opensource.org/licenses/mit-license.php).
