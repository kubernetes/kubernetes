// Copyright 2012 Google Inc.
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

goog.require('axs.AuditRule');

goog.provide('axs.AuditRules');

/**
 * Gets the audit rule with the given name.
 * @param {string} name
 * @return {axs.AuditRule}
 */
axs.AuditRules.getRule = function(name) {
    if (!axs.AuditRules.rules) {
        /** @type Object.<string, axs.AuditRule> */
        axs.AuditRules.rules = {};
        for (var specName in axs.AuditRule.specs) {
            var spec = axs.AuditRule.specs[specName];
            var auditRule = new axs.AuditRule(spec);
            axs.AuditRules.rules[spec.name] = auditRule;
        }
    }

    return axs.AuditRules.rules[name];
};
