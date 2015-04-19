// Copyright 2006 The Closure Library Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Constant declarations for common key codes.
 *
 * @author eae@google.com (Emil A Eklund)
 * @see ../demos/keyhandler.html
 */

goog.provide('goog.events.KeyCodes');

goog.require('goog.userAgent');


/**
 * Key codes for common characters.
 *
 * This list is not localized and therefore some of the key codes are not
 * correct for non US keyboard layouts. See comments below.
 *
 * @enum {number}
 */
goog.events.KeyCodes = {
  WIN_KEY_FF_LINUX: 0,
  MAC_ENTER: 3,
  BACKSPACE: 8,
  TAB: 9,
  NUM_CENTER: 12,  // NUMLOCK on FF/Safari Mac
  ENTER: 13,
  SHIFT: 16,
  CTRL: 17,
  ALT: 18,
  PAUSE: 19,
  CAPS_LOCK: 20,
  ESC: 27,
  SPACE: 32,
  PAGE_UP: 33,     // also NUM_NORTH_EAST
  PAGE_DOWN: 34,   // also NUM_SOUTH_EAST
  END: 35,         // also NUM_SOUTH_WEST
  HOME: 36,        // also NUM_NORTH_WEST
  LEFT: 37,        // also NUM_WEST
  UP: 38,          // also NUM_NORTH
  RIGHT: 39,       // also NUM_EAST
  DOWN: 40,        // also NUM_SOUTH
  PRINT_SCREEN: 44,
  INSERT: 45,      // also NUM_INSERT
  DELETE: 46,      // also NUM_DELETE
  ZERO: 48,
  ONE: 49,
  TWO: 50,
  THREE: 51,
  FOUR: 52,
  FIVE: 53,
  SIX: 54,
  SEVEN: 55,
  EIGHT: 56,
  NINE: 57,
  FF_SEMICOLON: 59, // Firefox (Gecko) fires this for semicolon instead of 186
  FF_EQUALS: 61, // Firefox (Gecko) fires this for equals instead of 187
  FF_DASH: 173, // Firefox (Gecko) fires this for dash instead of 189
  QUESTION_MARK: 63, // needs localization
  A: 65,
  B: 66,
  C: 67,
  D: 68,
  E: 69,
  F: 70,
  G: 71,
  H: 72,
  I: 73,
  J: 74,
  K: 75,
  L: 76,
  M: 77,
  N: 78,
  O: 79,
  P: 80,
  Q: 81,
  R: 82,
  S: 83,
  T: 84,
  U: 85,
  V: 86,
  W: 87,
  X: 88,
  Y: 89,
  Z: 90,
  META: 91, // WIN_KEY_LEFT
  WIN_KEY_RIGHT: 92,
  CONTEXT_MENU: 93,
  NUM_ZERO: 96,
  NUM_ONE: 97,
  NUM_TWO: 98,
  NUM_THREE: 99,
  NUM_FOUR: 100,
  NUM_FIVE: 101,
  NUM_SIX: 102,
  NUM_SEVEN: 103,
  NUM_EIGHT: 104,
  NUM_NINE: 105,
  NUM_MULTIPLY: 106,
  NUM_PLUS: 107,
  NUM_MINUS: 109,
  NUM_PERIOD: 110,
  NUM_DIVISION: 111,
  F1: 112,
  F2: 113,
  F3: 114,
  F4: 115,
  F5: 116,
  F6: 117,
  F7: 118,
  F8: 119,
  F9: 120,
  F10: 121,
  F11: 122,
  F12: 123,
  NUMLOCK: 144,
  SCROLL_LOCK: 145,

  // OS-specific media keys like volume controls and browser controls.
  FIRST_MEDIA_KEY: 166,
  LAST_MEDIA_KEY: 183,

  SEMICOLON: 186,            // needs localization
  DASH: 189,                 // needs localization
  EQUALS: 187,               // needs localization
  COMMA: 188,                // needs localization
  PERIOD: 190,               // needs localization
  SLASH: 191,                // needs localization
  APOSTROPHE: 192,           // needs localization
  TILDE: 192,                // needs localization
  SINGLE_QUOTE: 222,         // needs localization
  OPEN_SQUARE_BRACKET: 219,  // needs localization
  BACKSLASH: 220,            // needs localization
  CLOSE_SQUARE_BRACKET: 221, // needs localization
  WIN_KEY: 224,
  MAC_FF_META: 224, // Firefox (Gecko) fires this for the meta key instead of 91
  MAC_WK_CMD_LEFT: 91,  // WebKit Left Command key fired, same as META
  MAC_WK_CMD_RIGHT: 93, // WebKit Right Command key fired, different from META
  WIN_IME: 229,

  // We've seen users whose machines fire this keycode at regular one
  // second intervals. The common thread among these users is that
  // they're all using Dell Inspiron laptops, so we suspect that this
  // indicates a hardware/bios problem.
  // http://en.community.dell.com/support-forums/laptop/f/3518/p/19285957/19523128.aspx
  PHANTOM: 255
};


/**
 * Returns true if the event contains a text modifying key.
 * @param {goog.events.BrowserEvent} e A key event.
 * @return {boolean} Whether it's a text modifying key.
 */
goog.events.KeyCodes.isTextModifyingKeyEvent = function(e) {
  if (e.altKey && !e.ctrlKey ||
      e.metaKey ||
      // Function keys don't generate text
      e.keyCode >= goog.events.KeyCodes.F1 &&
      e.keyCode <= goog.events.KeyCodes.F12) {
    return false;
  }

  // The following keys are quite harmless, even in combination with
  // CTRL, ALT or SHIFT.
  switch (e.keyCode) {
    case goog.events.KeyCodes.ALT:
    case goog.events.KeyCodes.CAPS_LOCK:
    case goog.events.KeyCodes.CONTEXT_MENU:
    case goog.events.KeyCodes.CTRL:
    case goog.events.KeyCodes.DOWN:
    case goog.events.KeyCodes.END:
    case goog.events.KeyCodes.ESC:
    case goog.events.KeyCodes.HOME:
    case goog.events.KeyCodes.INSERT:
    case goog.events.KeyCodes.LEFT:
    case goog.events.KeyCodes.MAC_FF_META:
    case goog.events.KeyCodes.META:
    case goog.events.KeyCodes.NUMLOCK:
    case goog.events.KeyCodes.NUM_CENTER:
    case goog.events.KeyCodes.PAGE_DOWN:
    case goog.events.KeyCodes.PAGE_UP:
    case goog.events.KeyCodes.PAUSE:
    case goog.events.KeyCodes.PHANTOM:
    case goog.events.KeyCodes.PRINT_SCREEN:
    case goog.events.KeyCodes.RIGHT:
    case goog.events.KeyCodes.SCROLL_LOCK:
    case goog.events.KeyCodes.SHIFT:
    case goog.events.KeyCodes.UP:
    case goog.events.KeyCodes.WIN_KEY:
    case goog.events.KeyCodes.WIN_KEY_RIGHT:
      return false;
    case goog.events.KeyCodes.WIN_KEY_FF_LINUX:
      return !goog.userAgent.GECKO;
    default:
      return e.keyCode < goog.events.KeyCodes.FIRST_MEDIA_KEY ||
          e.keyCode > goog.events.KeyCodes.LAST_MEDIA_KEY;
  }
};


/**
 * Returns true if the key fires a keypress event in the current browser.
 *
 * Accoridng to MSDN [1] IE only fires keypress events for the following keys:
 * - Letters: A - Z (uppercase and lowercase)
 * - Numerals: 0 - 9
 * - Symbols: ! @ # $ % ^ & * ( ) _ - + = < [ ] { } , . / ? \ | ' ` " ~
 * - System: ESC, SPACEBAR, ENTER
 *
 * That's not entirely correct though, for instance there's no distinction
 * between upper and lower case letters.
 *
 * [1] http://msdn2.microsoft.com/en-us/library/ms536939(VS.85).aspx)
 *
 * Safari is similar to IE, but does not fire keypress for ESC.
 *
 * Additionally, IE6 does not fire keydown or keypress events for letters when
 * the control or alt keys are held down and the shift key is not. IE7 does
 * fire keydown in these cases, though, but not keypress.
 *
 * @param {number} keyCode A key code.
 * @param {number=} opt_heldKeyCode Key code of a currently-held key.
 * @param {boolean=} opt_shiftKey Whether the shift key is held down.
 * @param {boolean=} opt_ctrlKey Whether the control key is held down.
 * @param {boolean=} opt_altKey Whether the alt key is held down.
 * @return {boolean} Whether it's a key that fires a keypress event.
 */
goog.events.KeyCodes.firesKeyPressEvent = function(keyCode, opt_heldKeyCode,
    opt_shiftKey, opt_ctrlKey, opt_altKey) {
  if (!goog.userAgent.IE &&
      !(goog.userAgent.WEBKIT && goog.userAgent.isVersionOrHigher('525'))) {
    return true;
  }

  if (goog.userAgent.MAC && opt_altKey) {
    return goog.events.KeyCodes.isCharacterKey(keyCode);
  }

  // Alt but not AltGr which is represented as Alt+Ctrl.
  if (opt_altKey && !opt_ctrlKey) {
    return false;
  }

  // Saves Ctrl or Alt + key for IE and WebKit 525+, which won't fire keypress.
  // Non-IE browsers and WebKit prior to 525 won't get this far so no need to
  // check the user agent.
  if (goog.isNumber(opt_heldKeyCode)) {
    opt_heldKeyCode = goog.events.KeyCodes.normalizeKeyCode(opt_heldKeyCode);
  }
  if (!opt_shiftKey &&
      (opt_heldKeyCode == goog.events.KeyCodes.CTRL ||
       opt_heldKeyCode == goog.events.KeyCodes.ALT ||
       goog.userAgent.MAC &&
       opt_heldKeyCode == goog.events.KeyCodes.META)) {
    return false;
  }

  // Some keys with Ctrl/Shift do not issue keypress in WEBKIT.
  if (goog.userAgent.WEBKIT && opt_ctrlKey && opt_shiftKey) {
    switch (keyCode) {
      case goog.events.KeyCodes.BACKSLASH:
      case goog.events.KeyCodes.OPEN_SQUARE_BRACKET:
      case goog.events.KeyCodes.CLOSE_SQUARE_BRACKET:
      case goog.events.KeyCodes.TILDE:
      case goog.events.KeyCodes.SEMICOLON:
      case goog.events.KeyCodes.DASH:
      case goog.events.KeyCodes.EQUALS:
      case goog.events.KeyCodes.COMMA:
      case goog.events.KeyCodes.PERIOD:
      case goog.events.KeyCodes.SLASH:
      case goog.events.KeyCodes.APOSTROPHE:
      case goog.events.KeyCodes.SINGLE_QUOTE:
        return false;
    }
  }

  // When Ctrl+<somekey> is held in IE, it only fires a keypress once, but it
  // continues to fire keydown events as the event repeats.
  if (goog.userAgent.IE && opt_ctrlKey && opt_heldKeyCode == keyCode) {
    return false;
  }

  switch (keyCode) {
    case goog.events.KeyCodes.ENTER:
      return true;
    case goog.events.KeyCodes.ESC:
      return !goog.userAgent.WEBKIT;
  }

  return goog.events.KeyCodes.isCharacterKey(keyCode);
};


/**
 * Returns true if the key produces a character.
 * This does not cover characters on non-US keyboards (Russian, Hebrew, etc.).
 *
 * @param {number} keyCode A key code.
 * @return {boolean} Whether it's a character key.
 */
goog.events.KeyCodes.isCharacterKey = function(keyCode) {
  if (keyCode >= goog.events.KeyCodes.ZERO &&
      keyCode <= goog.events.KeyCodes.NINE) {
    return true;
  }

  if (keyCode >= goog.events.KeyCodes.NUM_ZERO &&
      keyCode <= goog.events.KeyCodes.NUM_MULTIPLY) {
    return true;
  }

  if (keyCode >= goog.events.KeyCodes.A &&
      keyCode <= goog.events.KeyCodes.Z) {
    return true;
  }

  // Safari sends zero key code for non-latin characters.
  if (goog.userAgent.WEBKIT && keyCode == 0) {
    return true;
  }

  switch (keyCode) {
    case goog.events.KeyCodes.SPACE:
    case goog.events.KeyCodes.QUESTION_MARK:
    case goog.events.KeyCodes.NUM_PLUS:
    case goog.events.KeyCodes.NUM_MINUS:
    case goog.events.KeyCodes.NUM_PERIOD:
    case goog.events.KeyCodes.NUM_DIVISION:
    case goog.events.KeyCodes.SEMICOLON:
    case goog.events.KeyCodes.FF_SEMICOLON:
    case goog.events.KeyCodes.DASH:
    case goog.events.KeyCodes.EQUALS:
    case goog.events.KeyCodes.FF_EQUALS:
    case goog.events.KeyCodes.COMMA:
    case goog.events.KeyCodes.PERIOD:
    case goog.events.KeyCodes.SLASH:
    case goog.events.KeyCodes.APOSTROPHE:
    case goog.events.KeyCodes.SINGLE_QUOTE:
    case goog.events.KeyCodes.OPEN_SQUARE_BRACKET:
    case goog.events.KeyCodes.BACKSLASH:
    case goog.events.KeyCodes.CLOSE_SQUARE_BRACKET:
      return true;
    default:
      return false;
  }
};


/**
 * Normalizes key codes from OS/Browser-specific value to the general one.
 * @param {number} keyCode The native key code.
 * @return {number} The normalized key code.
 */
goog.events.KeyCodes.normalizeKeyCode = function(keyCode) {
  if (goog.userAgent.GECKO) {
    return goog.events.KeyCodes.normalizeGeckoKeyCode(keyCode);
  } else if (goog.userAgent.MAC && goog.userAgent.WEBKIT) {
    return goog.events.KeyCodes.normalizeMacWebKitKeyCode(keyCode);
  } else {
    return keyCode;
  }
};


/**
 * Normalizes key codes from their Gecko-specific value to the general one.
 * @param {number} keyCode The native key code.
 * @return {number} The normalized key code.
 */
goog.events.KeyCodes.normalizeGeckoKeyCode = function(keyCode) {
  switch (keyCode) {
    case goog.events.KeyCodes.FF_EQUALS:
      return goog.events.KeyCodes.EQUALS;
    case goog.events.KeyCodes.FF_SEMICOLON:
      return goog.events.KeyCodes.SEMICOLON;
    case goog.events.KeyCodes.FF_DASH:
      return goog.events.KeyCodes.DASH;
    case goog.events.KeyCodes.MAC_FF_META:
      return goog.events.KeyCodes.META;
    case goog.events.KeyCodes.WIN_KEY_FF_LINUX:
      return goog.events.KeyCodes.WIN_KEY;
    default:
      return keyCode;
  }
};


/**
 * Normalizes key codes from their Mac WebKit-specific value to the general one.
 * @param {number} keyCode The native key code.
 * @return {number} The normalized key code.
 */
goog.events.KeyCodes.normalizeMacWebKitKeyCode = function(keyCode) {
  switch (keyCode) {
    case goog.events.KeyCodes.MAC_WK_CMD_RIGHT:  // 93
      return goog.events.KeyCodes.META;          // 91
    default:
      return keyCode;
  }
};
