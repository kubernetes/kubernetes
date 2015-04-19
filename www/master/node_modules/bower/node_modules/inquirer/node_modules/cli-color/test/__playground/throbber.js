#!/usr/bin/env node

'use strict';

var setupThrobber = require('../../throbber')

  , throbber = setupThrobber(process.stdout.write.bind(process.stdout), 200);

process.stdout.write('START');
throbber.start();
setTimeout(throbber.stop, 1100);
