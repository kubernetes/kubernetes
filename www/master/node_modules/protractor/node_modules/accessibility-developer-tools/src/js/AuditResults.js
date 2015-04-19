// Copyright 2013 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

goog.provide('axs.AuditResults');

/**
 * Object to hold results for an Audit run.
 * @constructor
 */
axs.AuditResults = function() {
  /**
   * The errors received from the audit run.
   * @type {Array.<string>}
   * @private
   */
  this.errors_ = [];

  /**
   * The warnings receive from the audit run.
   * @type {Array.<string>}
   * @private
   */
  this.warnings_ = [];
};
goog.exportSymbol('axs.AuditResults', axs.AuditResults);

/**
 * Adds an error message to the AuditResults object.
 * @param {string} errorMessage The error message to add.
 */
axs.AuditResults.prototype.addError = function(errorMessage) {
  if (errorMessage != '') {
    this.errors_.push(errorMessage);
  }

};
goog.exportProperty(axs.AuditResults.prototype, 'addError',
                    axs.AuditResults.prototype.addError);

/**
 * Adds a warning message to the AuditResults object.
 * @param {string} warningMessage The Warning message to add.
 */
axs.AuditResults.prototype.addWarning = function(warningMessage) {
  if (warningMessage != '') {
    this.warnings_.push(warningMessage);
  }

};
goog.exportProperty(axs.AuditResults.prototype, 'addWarning',
                    axs.AuditResults.prototype.addWarning);

/**
 * Returns the number of errors in this AuditResults object.
 * @return {number} The number of errors in the AuditResults object.
 */
axs.AuditResults.prototype.numErrors = function() {
  return this.errors_.length;
};
goog.exportProperty(axs.AuditResults.prototype, 'numErrors',
                    axs.AuditResults.prototype.numErrors);

/**
 * Returns the number of warnings in this AuditResults object.
 * @return {number} The number of warnings in the AuditResults object.
 */
axs.AuditResults.prototype.numWarnings = function() {
  return this.warnings_.length;
};
goog.exportProperty(axs.AuditResults.prototype, 'numWarnings',
                    axs.AuditResults.prototype.numWarnings);

/**
 * Returns the errors in this AuditResults object.
 * @return {Array.<string>} An array of the audit errors.
 */
axs.AuditResults.prototype.getErrors = function() {
  return this.errors_;
};
goog.exportProperty(axs.AuditResults.prototype, 'getErrors',
                    axs.AuditResults.prototype.getErrors);

/**
 * Returns the warnings in this AuditResults object.
 * @return {Array.<string>} An array of the audit warnings.
 */
axs.AuditResults.prototype.getWarnings = function() {
  return this.warnings_;
};
goog.exportProperty(axs.AuditResults.prototype, 'getWarnings',
                    axs.AuditResults.prototype.getWarnings);

/**
 * Returns a string message depicting AuditResults values.
 * @return {string} A string representation of the AuditResults object.
 */
axs.AuditResults.prototype.toString = function() {
  var message = '';
  for (var i = 0; i < this.errors_.length; i++) {
    if (i == 0) {
      message += '\nErrors:\n';
    }
    var result = this.errors_[i];
    message += result + '\n\n';
  }
  for (var i = 0; i < this.warnings_.length; i++) {
    if (i == 0) {
      message += '\nWarnings:\n';
    }
    var result = this.warnings_[i];
    message += result + '\n\n';
  }
  return message;
};
goog.exportProperty(axs.AuditResults.prototype, 'toString',
                    axs.AuditResults.prototype.toString);


