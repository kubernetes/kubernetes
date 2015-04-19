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

goog.provide('axs.constants');
goog.provide('axs.constants.Severity');
goog.provide('axs.constants.AuditResult');

/** @type {Object.<string, Object>} */
axs.constants.ARIA_ROLES = {
    "alert": {
        "namefrom": [ "author" ],
        "parent": [ "region" ]
    },
    "alertdialog": {
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "alert", "dialog" ]
    },
    "application": {
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "landmark" ]
    },
    "article": {
        "namefrom": [ "author" ],
        "parent": [ "document", "region" ]
    },
    "banner": {
        "namefrom": [ "author" ],
        "parent": [ "landmark" ]
    },
    "button": {
        "childpresentational": true,
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "command" ],
        "properties": [ "aria-expanded", "aria-pressed" ]
    },
    "checkbox": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "input" ],
        "requiredProperties": [ "aria-checked" ],
        "properties": [ "aria-checked" ]
    },
    "columnheader": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "gridcell", "sectionhead", "widget" ],
        "properties": [ "aria-sort" ]
    },
    "combobox": {
        "mustcontain": [ "listbox", "textbox" ],
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "select" ],
        "requiredProperties": [ "aria-expanded" ],
        "properties": [ "aria-expanded", "aria-autocomplete", "aria-required" ]
    },
    "command": {
        "abstract": true,
        "namefrom": [ "author" ],
        "parent": [ "widget" ]
    },
    "complementary": {
        "namefrom": [ "author" ],
        "parent": [ "landmark" ]
    },
    "composite": {
        "abstract": true,
        "childpresentational": false,
        "namefrom": [ "author" ],
        "parent": [ "widget" ],
        "properties": [ "aria-activedescendant" ]
    },
    "contentinfo": {
        "namefrom": [ "author" ],
        "parent": [ "landmark" ]
    },
    "definition": {
        "namefrom": [ "author" ],
        "parent": [ "section" ]
    },
    "dialog": {
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "window" ]
    },
    "directory": {
        "namefrom": [ "contents", "author" ],
        "parent": [ "list" ]
    },
    "document": {
        "namefrom": [ " author" ],
        "namerequired": true,
        "parent": [ "structure" ],
        "properties": [ "aria-expanded" ]
    },
    "form": {
        "namefrom": [ "author" ],
        "parent": [ "landmark" ]
    },
    "grid": {
        "mustcontain": [ "row", "rowgroup" ],
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "composite", "region" ],
        "properties": [ "aria-level", "aria-multiselectable", "aria-readonly" ]
    },
    "gridcell": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "section", "widget" ],
        "properties": [ "aria-readonly", "aria-required", "aria-selected" ]
    },
    "group": {
        "namefrom": [ " author" ],
        "parent": [ "section" ],
        "properties": [ "aria-activedescendant" ]
    },
    "heading": {
        "namerequired": true,
        "parent": [ "sectionhead" ],
        "properties": [ "aria-level" ]
    },
    "img": {
        "childpresentational": true,
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "section" ]
    },
    "input": {
        "abstract": true,
        "namefrom": [ "author" ],
        "parent": [ "widget" ]
    },
    "landmark": {
        "abstract": true,
        "namefrom": [ "contents", "author" ],
        "namerequired": false,
        "parent": [ "region" ]
    },
    "link": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "command" ],
        "properties": [ "aria-expanded" ]
    },
    "list": {
        "mustcontain": [ "group", "listitem" ],
        "namefrom": [ "author" ],
        "parent": [ "region" ]
    },
    "listbox": {
        "mustcontain": [ "option" ],
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "list", "select" ],
        "properties": [ "aria-multiselectable", "aria-required" ]
    },
    "listitem": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "section" ],
        "properties": [ "aria-level", "aria-posinset", "aria-setsize" ]
    },
    "log": {
        "namefrom": [ " author" ],
        "namerequired": true,
        "parent": [ "region" ]
    },
    "main": {
        "namefrom": [ "author" ],
        "parent": [ "landmark" ]
    },
    "marquee": {
        "namerequired": true,
        "parent": [ "section" ]
    },
    "math": {
        "childpresentational": true,
        "namefrom": [ "author" ],
        "parent": [ "section" ]
    },
    "menu": {
        "mustcontain": [
            "group",
            "menuitemradio",
            "menuitem",
            "menuitemcheckbox"
        ],
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "list", "select" ]
    },
    "menubar": {
        "namefrom": [ "author" ],
        "parent": [ "menu" ]
    },
    "menuitem": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "command" ]
    },
    "menuitemcheckbox": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "checkbox", "menuitem" ]
    },
    "menuitemradio": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "menuitemcheckbox", "radio" ]
    },
    "navigation": {
        "namefrom": [ "author" ],
        "parent": [ "landmark" ]
    },
    "note": {
        "namefrom": [ "author" ],
        "parent": [ "section" ]
    },
    "option": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "input" ],
        "properties": [
            "aria-checked",
            "aria-posinset",
            "aria-selected",
            "aria-setsize"
        ]
    },
    "presentation": {
        "parent": [ "structure" ]
    },
    "progressbar": {
        "childpresentational": true,
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "range" ]
    },
    "radio": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "checkbox", "option" ]
    },
    "radiogroup": {
        "mustcontain": [ "radio" ],
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "select" ],
        "properties": [ "aria-required" ]
    },
    "range": {
        "abstract": true,
        "namefrom": [ "author" ],
        "parent": [ "widget" ],
        "properties": [
            "aria-valuemax",
            "aria-valuemin",
            "aria-valuenow",
            "aria-valuetext"
        ]
    },
    "region": {
        "namefrom": [ " author" ],
        "parent": [ "section" ]
    },
    "roletype": {
        "abstract": true,
        "properties": [
            "aria-atomic",
            "aria-busy",
            "aria-controls",
            "aria-describedby",
            "aria-disabled",
            "aria-dropeffect",
            "aria-flowto",
            "aria-grabbed",
            "aria-haspopup",
            "aria-hidden",
            "aria-invalid",
            "aria-label",
            "aria-labelledby",
            "aria-live",
            "aria-owns",
            "aria-relevant"
        ]
    },
    "row": {
        "mustcontain": [ "columnheader", "gridcell", "rowheader" ],
        "namefrom": [ "contents", "author" ],
        "parent": [ "group", "widget" ],
        "properties": [ "aria-level", "aria-selected" ]
    },
    "rowgroup": {
        "mustcontain": [ "row" ],
        "namefrom": [ "contents", "author" ],
        "parent": [ "group" ]
    },
    "rowheader": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "gridcell", "sectionhead", "widget" ],
        "properties": [ "aria-sort" ]
    },
    "search": {
        "namefrom": [ "author" ],
        "parent": [ "landmark" ]
    },
    "section": {
        "abstract": true,
        "namefrom": [ "contents", "author" ],
        "parent": [ "structure" ],
        "properties": [ "aria-expanded" ]
    },
    "sectionhead": {
        "abstract": true,
        "namefrom": [ "contents", "author" ],
        "parent": [ "structure" ],
        "properties": [ "aria-expanded" ]
    },
    "select": {
        "abstract": true,
        "namefrom": [ "author" ],
        "parent": [ "composite", "group", "input" ]
    },
    "separator": {
        "childpresentational": true,
        "namefrom": [ "author" ],
        "parent": [ "structure" ],
        "properties": [ "aria-expanded", "aria-orientation" ]
    },
    "scrollbar": {
        "childpresentational": true,
        "namefrom": [ "author" ],
        "namerequired": false,
        "parent": [ "input", "range" ],
        "requiredProperties": [
            "aria-controls",
            "aria-orientation",
            "aria-valuemax",
            "aria-valuemin",
            "aria-valuenow"
        ],
        "properties": [
            "aria-controls",
            "aria-orientation",
            "aria-valuemax",
            "aria-valuemin",
            "aria-valuenow"
        ]
    },
    "slider": {
        "childpresentational": true,
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "input", "range" ],
        "requiredProperties": [ "aria-valuemax", "aria-valuemin", "aria-valuenow" ],
        "properties": [
            "aria-valuemax",
            "aria-valuemin",
            "aria-valuenow",
            "aria-orientation"
        ]
    },
    "spinbutton": {
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "input", "range" ],
        "requiredProperties": [ "aria-valuemax", "aria-valuemin", "aria-valuenow" ],
        "properties": [
            "aria-valuemax",
            "aria-valuemin",
            "aria-valuenow",
            "aria-required"
        ]
    },
    "status": {
        "parent": [ "region" ]
    },
    "structure": {
        "abstract": true,
        "parent": [ "roletype" ]
    },
    "tab": {
        "namefrom": [ "contents", "author" ],
        "parent": [ "sectionhead", "widget" ],
        "properties": [ "aria-selected" ]
    },
    "tablist": {
        "mustcontain": [ "tab" ],
        "namefrom": [ "author" ],
        "parent": [ "composite", "directory" ],
        "properties": [ "aria-level" ]
    },
    "tabpanel": {
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "region" ]
    },
    "textbox": {
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "input" ],
        "properties": [
            "aria-activedescendant",
            "aria-autocomplete",
            "aria-multiline",
            "aria-readonly",
            "aria-required"
        ]
    },
    "timer": {
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "status" ]
    },
    "toolbar": {
        "namefrom": [ "author" ],
        "parent": [ "group" ]
    },
    "tooltip": {
        "namerequired": true,
        "parent": [ "section" ]
    },
    "tree": {
        "mustcontain": [ "group", "treeitem" ],
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "select" ],
        "properties": [ "aria-multiselectable", "aria-required" ]
    },
    "treegrid": {
        "mustcontain": [ "row" ],
        "namefrom": [ "author" ],
        "namerequired": true,
        "parent": [ "grid", "tree" ]
    },
    "treeitem": {
        "namefrom": [ "contents", "author" ],
        "namerequired": true,
        "parent": [ "listitem", "option" ]
    },
    "widget": {
        "abstract": true,
        "parent": [ "roletype" ]
    },
    "window": {
        "abstract": true,
        "namefrom": [ " author" ],
        "parent": [ "roletype" ],
        "properties": [ "aria-expanded" ]
    }
};

axs.constants.WIDGET_ROLES = {};

/**
 * Squashes the parent hierarchy on to role object.
 * @param {Object} role
 * @param {Object} set
 * @private
 */
axs.constants.addAllParentRolesToSet_ = function(role, set) {
  if (!role['parent'])
      return;
  var parents = role['parent'];
  for (var j = 0; j < parents.length; j++) {
    var parentRoleName = parents[j];
    set[parentRoleName] = true;
    axs.constants.addAllParentRolesToSet_(
        axs.constants.ARIA_ROLES[parentRoleName], set);
  }
}

/**
 * Adds all properties and requiredProperties from parent hierarchy.
 * @param {Object} role
 * @param {string} propertiesName
 * @param {Object} propertiesSet
 * @private
 */
axs.constants.addAllPropertiesToSet_ = function(role, propertiesName,
    propertiesSet) {
  var properties = role[propertiesName]
  if (properties) {
    for (var i = 0; i < properties.length; i++)
      propertiesSet[properties[i]] = true;
  }
  if (role['parent']) {
    var parents = role['parent'];
    for (var j = 0; j < parents.length; j++) {
      var parentRoleName = parents[j];
      axs.constants.addAllPropertiesToSet_(
          axs.constants.ARIA_ROLES[parentRoleName], propertiesName,
          propertiesSet);
    }
  }
}

// TODO make a AriaRole object etc.
for (var roleName in axs.constants.ARIA_ROLES) {
    var role = axs.constants.ARIA_ROLES[roleName];

    var propertiesSet = {};
    axs.constants.addAllPropertiesToSet_(role, 'properties', propertiesSet);
    role['propertiesSet'] = propertiesSet;

    var requiredPropertiesSet = {};
    axs.constants.addAllPropertiesToSet_(role, 'requiredProperties', requiredPropertiesSet);
    role['requiredPropertiesSet'] = requiredPropertiesSet;
    var parentRolesSet = {};
    axs.constants.addAllParentRolesToSet_(role, parentRolesSet);
    role['allParentRolesSet'] = parentRolesSet;
    if ('widget' in parentRolesSet)
        axs.constants.WIDGET_ROLES[roleName] = role;
}

// BEGIN ARIA_PROPERTIES_AUTOGENERATED
/** @type {Object.<string, Object>} */
axs.constants.ARIA_PROPERTIES = {
    "activedescendant": {
        "type": "property",
        "valueType": "idref"
    },
    "atomic": {
        "defaultValue": "false",
        "type": "property",
        "valueType": "boolean"
    },
    "autocomplete": {
        "defaultValue": "none",
        "type": "property",
        "valueType": "token",
        "values": [
            "inline",
            "list",
            "both",
            "none"
        ]
    },
    "busy": {
        "defaultValue": "false",
        "type": "state",
        "valueType": "boolean"
    },
    "checked": {
        "defaultValue": "undefined",
        "type": "state",
        "valueType": "token",
        "values": [
            "true",
            "false",
            "mixed",
            "undefined"
        ]
    },
    "controls": {
        "type": "property",
        "valueType": "idref_list"
    },
    "describedby": {
        "type": "property",
        "valueType": "idref_list"
    },
    "disabled": {
        "defaultValue": "false",
        "type": "state",
        "valueType": "boolean"
    },
    "dropeffect": {
        "defaultValue": "none",
        "type": "property",
        "valueType": "token_list",
        "values": [
            "copy",
            "move",
            "link",
            "execute",
            "popup",
            "none"
        ]
    },
    "expanded": {
        "defaultValue": "undefined",
        "type": "state",
        "valueType": "token",
        "values": [
            "true",
            "false",
            "undefined"
        ]
    },
    "flowto": {
        "type": "property",
        "valueType": "idref_list"
    },
    "grabbed": {
        "defaultValue": "undefined",
        "type": "state",
        "valueType": "token",
        "values": [
            "true",
            "false",
            "undefined"
        ]
    },
    "haspopup": {
        "defaultValue": "false",
        "type": "property",
        "valueType": "boolean"
    },
    "hidden": {
        "defaultValue": "false",
        "type": "state",
        "valueType": "boolean"
    },
    "invalid": {
        "defaultValue": "false",
        "type": "state",
        "valueType": "token",
        "values": [
            "grammar",
            "false",
            "spelling",
            "true"
        ]
    },
    "label": {
        "type": "property",
        "valueType": "string"
    },
    "labelledby": {
        "type": "property",
        "valueType": "idref_list"
    },
    "level": {
        "type": "property",
        "valueType": "integer"
    },
    "live": {
        "defaultValue": "off",
        "type": "property",
        "valueType": "token",
        "values": [
            "off",
            "polite",
            "assertive"
        ]
    },
    "multiline": {
        "defaultValue": "false",
        "type": "property",
        "valueType": "boolean"
    },
    "multiselectable": {
        "defaultValue": "false",
        "type": "property",
        "valueType": "boolean"
    },
    "orientation": {
        "defaultValue": "vertical",
        "type": "property",
        "valueType": "token",
        "values": [
            "horizontal",
            "vertical"
        ]
    },
    "owns": {
        "type": "property",
        "valueType": "idref_list"
    },
    "posinset": {
        "type": "property",
        "valueType": "integer"
    },
    "pressed": {
        "defaultValue": "undefined",
        "type": "state",
        "valueType": "token",
        "values": [
            "true",
            "false",
            "mixed",
            "undefined"
        ]
    },
    "readonly": {
        "defaultValue": "false",
        "type": "property",
        "valueType": "boolean"
    },
    "relevant": {
        "defaultValue": "additions text",
        "type": "property",
        "valueType": "token_list",
        "values": [
            "additions",
            "removals",
            "text",
            "all"
        ]
    },
    "required": {
        "defaultValue": "false",
        "type": "property",
        "valueType": "boolean"
    },
    "selected": {
        "defaultValue": "undefined",
        "type": "state",
        "valueType": "token",
        "values": [
            "true",
            "false",
            "undefined"
        ]
    },
    "setsize": {
        "type": "property",
        "valueType": "integer"
    },
    "sort": {
        "defaultValue": "none",
        "type": "property",
        "valueType": "token",
        "values": [
            "ascending",
            "descending",
            "none",
            "other"
        ]
    },
    "valuemax": {
        "type": "property",
        "valueType": "decimal"
    },
    "valuemin": {
        "type": "property",
        "valueType": "decimal"
    },
    "valuenow": {
        "type": "property",
        "valueType": "decimal"
    },
    "valuetext": {
        "type": "property",
        "valueType": "string"
    }
};
// END ARIA_PROPERTIES_AUTOGENERATED

/**
 * All of the states and properties which apply globally.
 */
axs.constants.GLOBAL_PROPERTIES = [
    "aria-atomic",
    "aria-busy",  // (state)
    "aria-controls",
    "aria-describedby",
    "aria-disabled",  // (state)
    "aria-dropeffect",
    "aria-flowto",
    "aria-grabbed",  // (state)
    "aria-haspopup",
    "aria-hidden",  // (state)
    "aria-invalid",  // (state)
    "aria-label",
    "aria-labelledby",
    "aria-live",
    "aria-owns",
    "aria-relevant"
];

/**
 * A constant indicating no role name.
 * @type {string}
 */
axs.constants.NO_ROLE_NAME = ' ';

/**
 * A mapping from ARIA role names to their message ids.
 * Copied from ChromeVox:
 * http://code.google.com/p/google-axs-chrome/source/browse/trunk/chromevox/common/aria_util.js
 * @type {Object.<string, string>}
 */
axs.constants.WIDGET_ROLE_TO_NAME = {
  'alert' : 'aria_role_alert',
  'alertdialog' : 'aria_role_alertdialog',
  'button' : 'aria_role_button',
  'checkbox' : 'aria_role_checkbox',
  'columnheader' : 'aria_role_columnheader',
  'combobox' : 'aria_role_combobox',
  'dialog' : 'aria_role_dialog',
  'grid' : 'aria_role_grid',
  'gridcell' : 'aria_role_gridcell',
  'link' : 'aria_role_link',
  'listbox' : 'aria_role_listbox',
  'log' : 'aria_role_log',
  'marquee' : 'aria_role_marquee',
  'menu' : 'aria_role_menu',
  'menubar' : 'aria_role_menubar',
  'menuitem' : 'aria_role_menuitem',
  'menuitemcheckbox' : 'aria_role_menuitemcheckbox',
  'menuitemradio' : 'aria_role_menuitemradio',
  'option' : axs.constants.NO_ROLE_NAME,
  'progressbar' : 'aria_role_progressbar',
  'radio' : 'aria_role_radio',
  'radiogroup' : 'aria_role_radiogroup',
  'rowheader' : 'aria_role_rowheader',
  'scrollbar' : 'aria_role_scrollbar',
  'slider' : 'aria_role_slider',
  'spinbutton' : 'aria_role_spinbutton',
  'status' : 'aria_role_status',
  'tab' : 'aria_role_tab',
  'tabpanel' : 'aria_role_tabpanel',
  'textbox' : 'aria_role_textbox',
  'timer' : 'aria_role_timer',
  'toolbar' : 'aria_role_toolbar',
  'tooltip' : 'aria_role_tooltip',
  'treeitem' : 'aria_role_treeitem'
};


/**
 * @type {Object.<string, string>}
 * Copied from ChromeVox:
 * http://code.google.com/p/google-axs-chrome/source/browse/trunk/chromevox/common/aria_util.js
 */
axs.constants.STRUCTURE_ROLE_TO_NAME = {
  'article' : 'aria_role_article',
  'application' : 'aria_role_application',
  'banner' : 'aria_role_banner',
  'columnheader' : 'aria_role_columnheader',
  'complementary' : 'aria_role_complementary',
  'contentinfo' : 'aria_role_contentinfo',
  'definition' : 'aria_role_definition',
  'directory' : 'aria_role_directory',
  'document' : 'aria_role_document',
  'form' : 'aria_role_form',
  'group' : 'aria_role_group',
  'heading' : 'aria_role_heading',
  'img' : 'aria_role_img',
  'list' : 'aria_role_list',
  'listitem' : 'aria_role_listitem',
  'main' : 'aria_role_main',
  'math' : 'aria_role_math',
  'navigation' : 'aria_role_navigation',
  'note' : 'aria_role_note',
  'region' : 'aria_role_region',
  'rowheader' : 'aria_role_rowheader',
  'search' : 'aria_role_search',
  'separator' : 'aria_role_separator'
};


/**
 * @type {Array.<Object>}
 * Copied from ChromeVox:
 * http://code.google.com/p/google-axs-chrome/source/browse/trunk/chromevox/common/aria_util.js
 */
axs.constants.ATTRIBUTE_VALUE_TO_STATUS = [
  { name: 'aria-autocomplete', values:
      {'inline' : 'aria_autocomplete_inline',
       'list' : 'aria_autocomplete_list',
       'both' : 'aria_autocomplete_both'} },
  { name: 'aria-checked', values:
      {'true' : 'aria_checked_true',
       'false' : 'aria_checked_false',
       'mixed' : 'aria_checked_mixed'} },
  { name: 'aria-disabled', values:
      {'true' : 'aria_disabled_true'} },
  { name: 'aria-expanded', values:
      {'true' : 'aria_expanded_true',
       'false' : 'aria_expanded_false'} },
  { name: 'aria-invalid', values:
      {'true' : 'aria_invalid_true',
       'grammar' : 'aria_invalid_grammar',
       'spelling' : 'aria_invalid_spelling'} },
  { name: 'aria-multiline', values:
      {'true' : 'aria_multiline_true'} },
  { name: 'aria-multiselectable', values:
      {'true' : 'aria_multiselectable_true'} },
  { name: 'aria-pressed', values:
      {'true' : 'aria_pressed_true',
       'false' : 'aria_pressed_false',
       'mixed' : 'aria_pressed_mixed'} },
  { name: 'aria-readonly', values:
      {'true' : 'aria_readonly_true'} },
  { name: 'aria-required', values:
      {'true' : 'aria_required_true'} },
  { name: 'aria-selected', values:
      {'true' : 'aria_selected_true',
       'false' : 'aria_selected_false'} }
];

/**
 * Copied from ChromeVox:
 * http://code.google.com/p/google-axs-chrome/source/browse/trunk/chromevox/common/dom_util.js
 * @type {Object}
 */
axs.constants.INPUT_TYPE_TO_INFORMATION_TABLE_MSG = {
  'button' : 'input_type_button',
  'checkbox' : 'input_type_checkbox',
  'color' : 'input_type_color',
  'datetime' : 'input_type_datetime',
  'datetime-local' : 'input_type_datetime_local',
  'date' : 'input_type_date',
  'email' : 'input_type_email',
  'file' : 'input_type_file',
  'image' : 'input_type_image',
  'month' : 'input_type_month',
  'number' : 'input_type_number',
  'password' : 'input_type_password',
  'radio' : 'input_type_radio',
  'range' : 'input_type_range',
  'reset' : 'input_type_reset',
  'search' : 'input_type_search',
  'submit' : 'input_type_submit',
  'tel' : 'input_type_tel',
  'text' : 'input_type_text',
  'url' : 'input_type_url',
  'week' : 'input_type_week'
};


/**
 * Copied from ChromeVox:
 * http://code.google.com/p/google-axs-chrome/source/browse/trunk/chromevox/common/dom_util.js
 * @type {Object}
 */
axs.constants.TAG_TO_INFORMATION_TABLE_VERBOSE_MSG = {
  'A' : 'tag_link',
  'BUTTON' : 'tag_button',
  'H1' : 'tag_h1',
  'H2' : 'tag_h2',
  'H3' : 'tag_h3',
  'H4' : 'tag_h4',
  'H5' : 'tag_h5',
  'H6' : 'tag_h6',
  'LI' : 'tag_li',
  'OL' : 'tag_ol',
  'SELECT' : 'tag_select',
  'TEXTAREA' : 'tag_textarea',
  'UL' : 'tag_ul',
  'SECTION' : 'tag_section',
  'NAV' : 'tag_nav',
  'ARTICLE' : 'tag_article',
  'ASIDE' : 'tag_aside',
  'HGROUP' : 'tag_hgroup',
  'HEADER' : 'tag_header',
  'FOOTER' : 'tag_footer',
  'TIME' : 'tag_time',
  'MARK' : 'tag_mark'
};

/**
 * Copied from ChromeVox:
 * http://code.google.com/p/google-axs-chrome/source/browse/trunk/chromevox/common/dom_util.js
 * @type {Object}
 */
axs.constants.TAG_TO_INFORMATION_TABLE_BRIEF_MSG = {
  'BUTTON' : 'tag_button',
  'SELECT' : 'tag_select',
  'TEXTAREA' : 'tag_textarea'
};

axs.constants.MIXED_VALUES = {
    "true": true,
    "false": true,
    "mixed": true
};

(function() {
// pull values lists into sets
for (var propertyName in axs.constants.ARIA_PROPERTIES) {
    var propertyDetails = axs.constants.ARIA_PROPERTIES[propertyName];
    if (!propertyDetails.values)
        continue;
    var valuesSet = {};
    for (var i = 0; i < propertyDetails.values.length; i++)
        valuesSet[propertyDetails.values[i]] = true;
    propertyDetails.valuesSet = valuesSet;
}
})();

/** @enum {string} */
axs.constants.Severity =  {
    INFO: 'Info',
    WARNING: 'Warning',
    SEVERE: 'Severe'
};

/** @enum {string} */
axs.constants.AuditResult = {
    PASS: 'PASS',
    FAIL: 'FAIL',
    NA: 'NA'
};

/** @enum {boolean} */
axs.constants.InlineElements = {
    // fontstyle
    'TT': true,
    'I': true,
    'B': true,
    'BIG': true,
    'SMALL': true,

    // phrase
    'EM': true,
    'STRONG': true,
    'DFN': true,
    'CODE': true,
    'SAMP': true,
    'KBD': true,
    'VAR': true,
    'CITE': true,
    'ABBR': true,
    'ACRONYM': true,

    // special
    'A': true,
    'IMG': true,
    'OBJECT': true,
    'BR': true,
    'SCRIPT': true,
    'MAP': true,
    'Q': true,
    'SUB': true,
    'SUP': true,
    'SPAN': true,
    'BDO': true,

    // formctrl
    'INPUT': true,
    'SELECT': true,
    'TEXTAREA': true,
    'LABEL': true,
    'BUTTON': true
 }
