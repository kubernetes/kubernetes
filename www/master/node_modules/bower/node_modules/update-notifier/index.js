'use strict';
var spawn = require('child_process').spawn;
var path = require('path');
var Configstore = require('configstore');
var chalk = require('chalk');
var semverDiff = require('semver-diff');
var latestVersion = require('latest-version');
var stringLength = require('string-length');
var isNpm = require('is-npm');

function UpdateNotifier(options) {
	this.options = options = options || {};
	options.pkg = options.pkg || {};

	// deprecated options
	// TODO: remove this at some point far into the future
	if (options.packageName && options.packageVersion) {
		options.pkg.name = options.packageName;
		options.pkg.version = options.packageVersion;
	}

	if (!options.pkg.name || !options.pkg.version) {
		throw new Error('pkg.name and pkg.version required');
	}

	this.packageName = options.pkg.name;
	this.packageVersion = options.pkg.version;
	this.updateCheckInterval = typeof options.updateCheckInterval === 'number' ? options.updateCheckInterval : 1000 * 60 * 60 * 24; // 1 day
	this.hasCallback = typeof options.callback === 'function';
	this.callback = options.callback || function () {};

	if (!this.hasCallback) {
		this.config = new Configstore('update-notifier-' + this.packageName, {
			optOut: false,
			// init with the current time so the first check is only
			// after the set interval, so not to bother users right away
			lastUpdateCheck: Date.now()
		});
	}
}

UpdateNotifier.prototype.check = function () {
	if (this.hasCallback) {
		return this.checkNpm(this.callback);
	}

	if (this.config.get('optOut') || 'NO_UPDATE_NOTIFIER' in process.env) {
		return;
	}

	this.update = this.config.get('update');

	if (this.update) {
		this.config.del('update');
	}

	// Only check for updates on a set interval
	if (Date.now() - this.config.get('lastUpdateCheck') < this.updateCheckInterval) {
		return;
	}

	// Spawn a detached process, passing the options as an environment property
	spawn(process.execPath, [path.join(__dirname, 'check.js'), JSON.stringify(this.options)], {
		detached: true,
		stdio: 'ignore'
	}).unref();
};

UpdateNotifier.prototype.checkNpm = function (cb) {
	latestVersion(this.packageName, function (err, latestVersion) {
		if (err) {
			return cb(err);
		}

		cb(null, {
			latest: latestVersion,
			current: this.packageVersion,
			type: semverDiff(this.packageVersion, latestVersion) || 'latest',
			name: this.packageName
		});
	}.bind(this));
};

UpdateNotifier.prototype.notify = function (opts) {
	if (!process.stdout.isTTY || isNpm || !this.update) {
		return this;
	}

	opts = opts || {};
	opts.defer = opts.defer === undefined ? true : false;

	var fill = function (str, count) {
		return Array(count + 1).join(str);
	};

	var line1 = ' Update available: ' + chalk.green.bold(this.update.latest) +
		chalk.dim(' (current: ' + this.update.current + ')') + ' ';
	var line2 = ' Run ' + chalk.blue('npm install -g ' + this.packageName) +
		' to update. ';
	var contentWidth = Math.max(stringLength(line1), stringLength(line2));
	var line1rest = contentWidth - stringLength(line1);
	var line2rest = contentWidth - stringLength(line2);
	var top = chalk.yellow('┌' + fill('─', contentWidth) + '┐');
	var bottom = chalk.yellow('└' + fill('─', contentWidth) + '┘');
	var side = chalk.yellow('│');

	var message =
		'\n\n' +
		top + '\n' +
		side + line1 + fill(' ', line1rest) + side + '\n' +
		side + line2 + fill(' ', line2rest) + side + '\n' +
		bottom + '\n';

	if (opts.defer) {
		process.on('exit', function () {
			console.error(message);
		});
	} else {
		console.error(message);
	}

	return this;
};

module.exports = function (options) {
	var updateNotifier = new UpdateNotifier(options);
	updateNotifier.check();
	return updateNotifier;
};

module.exports.UpdateNotifier = UpdateNotifier;
