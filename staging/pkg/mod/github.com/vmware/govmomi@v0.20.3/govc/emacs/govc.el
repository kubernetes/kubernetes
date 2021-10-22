;;; govc.el --- Interface to govc for managing VMware ESXi and vCenter

;; Author: The govc developers
;; URL: https://github.com/vmware/govmomi/tree/master/govc/emacs
;; Keywords: convenience
;; Version: 0.18.0
;; Package-Requires: ((emacs "24.3") (dash "1.5.0") (s "1.9.0") (magit-popup "2.0.50") (json-mode "1.6.0"))

;; This file is NOT part of GNU Emacs.

;; Copyright (c) 2016 VMware, Inc. All Rights Reserved.
;;
;; Licensed under the Apache License, Version 2.0 (the "License");
;; you may not use this file except in compliance with the License.
;; You may obtain a copy of the License at
;;
;; http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.

;;; Commentary:

;; The goal of this package is to provide a simple interface for commonly used
;; govc commands within Emacs.  This includes table based inventory/state modes
;; for vms, hosts, datastores and pools.  The keymap for each mode provides
;; shortcuts for easily feeding the data in view to other govc commands.
;;
;; Within the various govc modes, press `?' to see a popup menu of options.
;; A menu bar is enabled for certain modes, such as `govc-vm-mode' and `govc-host-mode'.
;; There is also a `govc' menu at all times under the `Tools' menu.
;;
;; The recommended way to install govc.el is via MELPA (http://melpa.org/).

;;; Code:

(eval-when-compile
  (require 'cl))
(require 'dash)
(require 'diff)
(require 'dired)
(require 'json-mode)
(require 'magit-popup)
(require 'url-parse)
(require 's)

(autoload 'auth-source-search "auth-source")

(defgroup govc nil
  "Emacs customization group for govc."
  :group 'convenience)

(defcustom govc-keymap-prefix "C-c ;"
  "Prefix for `govc-mode'."
  :group 'govc)

(defcustom govc-command "govc"
  "Executable path to the govc utility."
  :type 'string
  :group 'govc)

(defvar govc-command-map
  (let ((map (make-sparse-keymap)))
    (define-key map "h" 'govc-host)
    (define-key map "p" 'govc-pool)
    (define-key map "v" 'govc-vm)
    (define-key map "s" 'govc-datastore)
    (define-key map "?" 'govc-popup)
    map)
  "Keymap for `govc-mode' after `govc-keymap-prefix' was pressed.")

(defvar govc-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd govc-keymap-prefix) govc-command-map)
    map)
  "Keymap for `govc-mode'.")

;;;###autoload
(define-minor-mode govc-mode
  "Running `govc-global-mode' creates key bindings to the various govc modes.
The default prefix is `C-c ;' and can be changed by setting `govc-keymap-prefix'.

\\{govc-mode-map\}"
  nil govc-mode-line govc-mode-map
  :group 'govc)

;;;###autoload
(define-globalized-minor-mode govc-global-mode govc-mode govc-mode)

(defcustom govc-mode-line
  '(:eval (format " govc[%s]" (or (govc-session-name) "-")))
  "Mode line lighter for govc."
  :group 'govc
  :type 'sexp
  :risky t)


;;; Tabulated list mode extensions (derived from https://github.com/Silex/docker.el tabulated-list-ext.el)
(defun govc-tabulated-list-mark ()
  "Mark and move to the next line."
  (interactive)
  (tabulated-list-put-tag (char-to-string dired-marker-char) t))

(defun govc-tabulated-list-unmark ()
  "Unmark and move to the next line."
  (interactive)
  (tabulated-list-put-tag "" t))

(defun govc-tabulated-list-toggle-marks ()
  "Toggle mark."
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (let ((cmd))
      (while (not (eobp))
        (setq cmd (char-after))
        (tabulated-list-put-tag
         (if (eq cmd dired-marker-char)
             ""
           (char-to-string dired-marker-char)) t)))))

(defun govc-tabulated-list-unmark-all ()
  "Unmark all."
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (while (not (eobp))
      (tabulated-list-put-tag "" t))))

(defun govc-selection ()
  "Get the current selection as a list of names."
  (let ((selection))
    (save-excursion
      (goto-char (point-min))
      (while (not (eobp))
        (when (eq (char-after) ?*)
          (add-to-list 'selection (tabulated-list-get-id)))
        (forward-line)))
    (or selection (let ((id (tabulated-list-get-id)))
                    (if id
                        (list id))))))

(defun govc-do-selection (fn action)
  "Call FN with `govc-selection' confirming ACTION."
  (let* ((selection (govc-selection))
         (count (length selection))
         (prompt (if (= count 1)
                     (car selection)
                   (format "* [%d] marked" count))))
    (if (yes-or-no-p (format "%s %s ?" action prompt))
        (funcall fn selection))))

(defun govc-copy-selection ()
  "Copy current selection or region to the kill ring."
  (interactive)
  (if (region-active-p)
      (copy-region-as-kill (mark) (point) 'region)
    (kill-new (message "%s" (s-join " " (--map (format "\"%s\"" it) (govc-selection)))))))

(defvar govc-font-lock-keywords
  `((,(let ((host-expression "\\b[-a-z0-9]+\\b")) ;; Hostname
        (concat
         (mapconcat 'identity (make-list 3 host-expression) "\\.")
         "\\(\\." host-expression "\\)*")) .
         (0 font-lock-variable-name-face))
    (,(mapconcat 'identity (make-list 4 "[0-9]+") "\\.") ;; IP address
     . (0 font-lock-variable-name-face))
    ("\"[^\"]*\"" . (0 font-lock-string-face))
    ("'[^']*'" . (0 font-lock-string-face))
    ("[.0-9]+%" . (0 font-lock-type-face))
    ("\\<\\(success\\|poweredOn\\)\\>" . (1 font-lock-doc-face))
    ("\\<\\(error\\|poweredOff\\)\\>" . (1 font-lock-warning-face))
    ("\\<\\(running\\|info\\)\\>" . (1 font-lock-variable-name-face))
    ("\\<\\(warning\\|suspended\\)\\>" . (1 font-lock-keyword-face))
    ("\\<\\(verbose\\|trivia\\)\\>" . (1 whitespace-line))
    (,dired-re-maybe-mark . (0 dired-mark-face))
    ("types.ManagedObjectReference\\(.*\\)" . (1 dired-directory-face))
    ("[^ ]*/$" . (0 dired-directory-face))
    ("\\.\\.\\.$" . (0 dired-symlink-face))
    ("^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9][A-Z][0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][A-Z]|" . (0 font-lock-comment-face))
    ("^\\([ A-Za-z0-9_]+: \\).*" . (1 font-lock-comment-face))
    ("<[^>]*>" . (0 font-lock-comment-face))
    ("\\[[^]]*\\]" . (0 font-lock-comment-face))
    ("([^)]*)" . (0 font-lock-comment-face))))

(defvar govc-tabulated-list-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "m" 'govc-tabulated-list-mark)
    (define-key map "u" 'govc-tabulated-list-unmark)
    (define-key map "t" 'govc-tabulated-list-toggle-marks)
    (define-key map "U" 'govc-tabulated-list-unmark-all)
    (define-key map (kbd "M-&") 'govc-shell-command)
    (define-key map (kbd "M-w") 'govc-copy-selection)
    (define-key map (kbd "M-E") 'govc-copy-environment)
    map)
  "Keymap for `govc-tabulated-list-mode'.")

(define-derived-mode govc-tabulated-list-mode tabulated-list-mode "Tabulated govc"
  "Generic table bindings to mark/unmark rows."
  (setq-local font-lock-defaults
              '(govc-font-lock-keywords t nil nil beginning-of-line)))


;;; Keymap helpers for generating menus and popups
(defun govc-keymap-list (keymap)
  "Return a list of (key name function) for govc bindings in the given KEYMAP.
The name returned is the first word of the function `documentation'."
  (let ((map))
    (map-keymap
     (lambda (k f)
       (when (keymapp f)
         (setq map (append map
                           (--map (and (setcar it (kbd (format "M-%s" (char-to-string (car it))))) it)
                                  (govc-keymap-list f)))))
       (when (and (symbolp f)
                  (s-starts-with? "govc-" (symbol-name f)))
         (if (not (eq ?? k))
             (add-to-list 'map (list k (car (split-string (documentation f))) f))))) keymap)
    map))

(defun govc-keymap-menu (keymap)
  "Return a list of [key function t] for govc bindings in the given KEYMAP.
For use with `easy-menu-define'."
  (-map (lambda (item)
          (vector (nth 1 item) (nth 2 item) t))
        (govc-keymap-list keymap)))

(defun govc-key-description (key)
  "Call `key-description' ensuring KEY is a sequence."
  (key-description (if (integerp key) (list key) key)))

(defun govc-keymap-list-to-help (keymap)
  "Convert KEYMAP to list of help text."
  (--map (list (govc-key-description (car it))
               (car (split-string (documentation (nth 2 it)) "\\.")))
         keymap))

(defun govc-keymap-popup-help ()
  "Default keymap help for `govc-keymap-popup'."
  (append (govc-keymap-list-to-help (govc-keymap-list govc-tabulated-list-mode-map))
          '(("g" "Refresh current buffer")
            ("C-h m" "Show all key bindings"))))

(defun govc-keymap-popup (keymap)
  "Convert a `govc-keymap-list' using KEYMAP for use with `magit-define-popup'.
Keys in the ASCII range of 32-97 are mapped to popup commands, all others are listed as help text."
  (let* ((maps (--separate (and (integerp (car it))
                                (>= (car it) 32)
                                (<= (car it) 97))
                           (govc-keymap-list keymap)))
         (help (govc-keymap-list-to-help (cadr maps))))
    (append
     '("Commands")
     (car maps)
     (list (s-join "\n" (--map (format " %-6s %s" (car it) (cadr it))
                               (append help (govc-keymap-popup-help))))
           nil))))


;;; govc process helpers
(defcustom govc-urls nil
  "List of URLs for use with `govc-session'.
The `govc-session-name' displayed by `govc-mode-line' uses `url-target' (anchor)
if set, otherwise `url-host' is used.

Example:
```
  (setq govc-urls '(\"root:vagrant@localhost:18443#Vagrant-ESXi\"
                    \"root:password@192.168.1.192#Intel-NUC\"
                    \"Administrator@vsphere.local:password!@vcva-clovervm\"))
```
To enter a URL that is not in the list, prefix `universal-argument', for example:

  `\\[universal-argument] \\[govc-vm]'

To avoid putting your credentials in a variable, you can use the
auth-source search integration.

```
  (setq govc-urls '(\"myserver-vmware-2\"))
```

And then put this line in your `auth-sources' (e.g. `~/.authinfo.gpg'):
```
    machine myserver-vmware-2 login tzz password mypass url \"myserver-vmware-2.some.domain.here:443?insecure=true\"
```

Which will result in the URL \"tzz:mypass@myserver-vmware-2.some.domain.here:443?insecure=true\".
For more details on `auth-sources', see Info node `(auth) Help for users'.

When in `govc-vm' or `govc-host' mode, a default URL is composed with the
current session credentials and the IP address of the current vm/host and
the vm/host name as the session name.  This makes it easier to connect to
nested ESX/vCenter VMs or directly to an ESX host."
  :group 'govc
  :type '(repeat (string :tag "vcenter URL or auth-source machine reference")))

(defvar-local govc-session-url nil
  "ESX or vCenter URL set by `govc-session' via `govc-urls' selection.")

(defvar-local govc-session-path nil)

(defvar-local govc-session-insecure nil
  "Skip verification of server certificate when true.
This variable is set to the value of the `GOVC_INSECURE' env var by default.
It can also be set per-url via the query string (insecure=true).  For example:
```
  (setq govc-urls '(\"root:password@hostname?insecure=true\"))
```")

(defvar-local govc-session-datacenter nil
  "Datacenter to use for the current `govc-session'.
If the endpoint has a single Datacenter it will be used by default, otherwise
`govc-session' will prompt for selection.  It can also be set per-url via the
query string.  For example:
```
  (setq govc-urls '(\"root:password@hostname?datacenter=dc1\"))
```")

(defvar-local govc-session-datastore nil
  "Datastore to use for the current `govc-session'.
If the endpoint has a single Datastore it will be used by default, otherwise
`govc-session' will prompt for selection.  It can also be set per-url via the
query string.  For example:
```
  (setq govc-urls '(\"root:password@hostname?datastore=vsanDatastore\"))
```")

(defvar-local govc-session-network nil
  "Network to use for the current `govc-session'.")

(defvar-local govc-filter nil
  "Resource path filter.")

(defvar-local govc-args nil
  "Additional govc arguments.")

(defun govc-session-name ()
  "Return a name for the current session.
Derived from `govc-session-url' if set, otherwise from the 'GOVC_URL' env var.
Return value is the url anchor if set, otherwise the hostname is returned."
  (let* ((u (or govc-session-url (getenv "GOVC_URL")))
         (url (if u (govc-url-parse u))))
    (if url
        (concat (or (url-target url) (url-host url)) govc-session-path))))

(defun govc-format-command (command &rest args)
  "Format govc COMMAND ARGS."
  (format "%s %s %s" govc-command command
          (s-join " " (--map (format "\"%s\"" it)
                             (-flatten (-non-nil args))))))

(defconst govc-environment-map (--map (cons (concat "GOVC_" (upcase it))
                                            (intern (concat "govc-session-" it)))
                                      '("url" "insecure" "datacenter" "datastore" "network"))

  "Map of `GOVC_*' environment variable names to `govc-session-*' symbol names.")

(defun govc-environment (&optional unset)
  "Return `process-environment' for govc.
Optionally clear govc env if UNSET is non-nil."
  (let ((process-environment (copy-sequence process-environment)))
    (dolist (e govc-environment-map)
      (setenv (car e) (unless unset (symbol-value (cdr e)))))
    process-environment))

(defun govc-export-environment (arg)
  "Set if ARG is \\[universal-argument], unset if ARG is \\[negative-argument]."
  (if (equal arg '-)
      (progn (setq process-environment (govc-environment t))
             (cons "unset" (--map (car it)
                                  govc-environment-map)))
    (progn (setq process-environment (govc-environment))
           (cons "export" (--map (format "%s='%s'" (car it) (or (symbol-value (cdr it)) ""))
                                 govc-environment-map)))))

(defun govc-copy-environment (&optional arg)
  "Export session to `process-environment' and `kill-ring'.
Optionally set `GOVC_*' vars in `process-environment' using prefix
\\[universal-argument] ARG or unset with prefix \\[negative-argument] ARG."
  (interactive "P")
  (message (kill-new (if arg (s-join " " (govc-export-environment arg)) govc-session-url))))

(defun govc-process (command handler)
  "Run COMMAND, calling HANDLER upon successful exit of the process."
  (message "%s" command)
  (let ((process-environment (govc-environment))
        (exit-code))
    (add-to-list 'govc-command-history command)
    (with-temp-buffer
      (setq exit-code (call-process-shell-command command nil (current-buffer)))
      (if (zerop exit-code)
          (funcall handler)
        (error (buffer-string))))))

(defun govc (command &rest args)
  "Execute govc COMMAND with ARGS.
Return value is `buffer-string' split on newlines."
  (govc-process (govc-format-command command args)
                (lambda ()
                  (split-string (buffer-string) "\n" t))))

(defun govc-json (command &rest args)
  "Execute govc COMMAND passing arguments ARGS.
Return value is `json-read'."
  (govc-process (govc-format-command command (cons "-json" args))
                (lambda ()
                  (goto-char (point-min))
                  (let ((json-object-type 'plist))
                    (json-read)))))

(defun govc-ls-datacenter ()
  "List datacenters."
  (govc "ls" "-t" "Datacenter" "./..."))

(defun govc-object-prompt (prompt ls)
  "PROMPT for object name via LS function.  Return object without PROMPT if there is just one instance."
  (let ((objs (if (listp ls) ls (funcall ls))))
    (if (eq 1 (length objs))
        (car objs)
      (completing-read prompt objs))))

(defun govc-url-parse (url)
  "A `url-generic-parse-url' wrapper to handle URL with password, but no scheme.
Also fixes the case where user contains an '@'."
  (let* ((full (s-contains? "://" url))
         (u (url-generic-parse-url (concat (unless full "https://") url))))
    (unless full
      (setf (url-type u) nil)
      (setf (url-fullness u) nil))
    (if (s-contains? "@" (url-host u))
        (let* ((h (split-string (url-host u) "@"))
               (p (split-string (car h) ":")))
          (setf (url-host u) (cadr h))
          (setf (url-user u) (concat (url-user u) "@" (car p)))
          (setf (url-password u) (cadr p))))
    u))

(defun govc-url-default ()
  "Default URL when creating a new session."
  (if govc-session-url
      (let ((url (govc-url-parse govc-session-url)))
        (if (equal major-mode 'govc-host-mode)
            (progn (setf (url-host url) (govc-table-column-value "Name"))
                   (setf (url-target url) nil))
          (progn (setf (url-host url) (govc-table-column-value "IP address"))
                 (setf (url-target url) (govc-table-column-value "Name"))
                 ;; default url-user to Administrator@$domain when connecting to a vCenter VM
                 (let ((sts (ignore-errors (govc "sso.service.ls" "-t" "sso:sts" "-U" "-u" (url-host url)))))
                   (if sts (setf (url-user url) (concat "Administrator@" (file-name-nondirectory (car sts))))))))
        (setf (url-filename url) "") ; erase query string
        (if (string-empty-p (url-user url))
            (setf (url-user url) "root")) ; local workstation url has no user set
        (url-recreate-url url))))

(defun govc-urls-completing-read ()
  "A wrapper for `completing-read' to mask credentials in `govc-urls'."
  (let ((alist))
    (dolist (ent govc-urls)
      (let ((u (govc-url-parse ent)))
        (setf (url-password u) nil)
        (add-to-list 'alist `(,(url-recreate-url u) . ,ent) t)))
    (let ((u (completing-read "govc url: " (-map 'car alist))))
      (cdr (assoc u alist)))))

(defun govc-session-url-lookup-auth-source (url-or-address)
  "Check if URL-OR-ADDRESS is a logical name in the authinfo file.
Given URL-OR-ADDRESS `myserver-vmware-2' this function will find
a line like
    machine myserver-vmware-2 login tzz password mypass url \"myserver-vmware-2.some.domain.here:443?insecure=true\"

and will return the URL \"tzz:mypass@myserver-vmware-2.some.domain.here:443?insecure=true\".

If the line is not found, the original URL-OR-ADDRESS is
returned, assuming that's what the user wanted."
  (let ((found (nth 0 (auth-source-search :max 1
                                          :host url-or-address
                                          :require '(:user :secret :url)
                                          :create nil))))
    (if found
        (format "%s:%s@%s"
                (plist-get found :user)
                (let ((secret (plist-get found :secret)))
                  (if (functionp secret)
                      (funcall secret)
                    secret))
                (plist-get found :url))
      url-or-address)))

(defun govc-session-set-url (url)
  "Set `govc-session-url' to URL and optionally set other govc-session-* variables via URL query."
  ;; Replace the original URL with the auth-source lookup if there is no user.
  (unless (url-user (govc-url-parse url))
    (setq url (govc-session-url-lookup-auth-source url)))

  (let ((q (cdr (url-path-and-query (govc-url-parse url)))))
    (dolist (opt (if q (url-parse-query-string q)))
      (let ((var (intern (concat "govc-session-" (car opt)))))
        (if (boundp var)
            (set var (cadr opt))))))
  (setq govc-session-url url))

(defun govc-session ()
  "Initialize a govc session."
  (interactive)
  (let ((url (if (or current-prefix-arg (eq 0 (length govc-urls)))
                 (read-string "govc url: " (govc-url-default))
               (if (eq 1 (length govc-urls))
                   (car govc-urls)
                 (govc-urls-completing-read)))))
    ;; Wait until this point to clear so current session is preserved in the
    ;; event of `keyboard-quit' in `read-string'.
    (setq govc-session-datacenter nil
          govc-session-datastore nil
          govc-session-network nil
          govc-filter nil)
    (govc-session-set-url url))
  (unless govc-session-insecure
    (setq govc-session-insecure (or (getenv "GOVC_INSECURE")
                                    (completing-read "govc insecure: " '("true" "false")))))
  (unless govc-session-datacenter
    (setq govc-session-datacenter (govc-object-prompt "govc datacenter: " 'govc-ls-datacenter)))
  (add-to-list 'govc-urls govc-session-url))

(defalias 'govc-current-session 'buffer-local-variables)

(defun govc-session-clone (session)
  "Clone a session from SESSION buffer locals."
  (dolist (v session)
    (let ((s (car v)))
      (when (s-starts-with? "govc-session-" (symbol-name s))
        (set s (assoc-default s session))))))

(defvar govc-command-history nil
  "History list for govc commands used by `govc-shell-command'.")

(defvar govc-shell--revert-cmd nil)

(defun govc-shell--revert-function (&optional _ _)
  "Re-run the buffer's most recent govc-shell-run command."
  (apply (car govc-shell--revert-cmd) (cdr govc-shell--revert-cmd)))

(defun govc-shell-filter (proc string)
  "Process filter for govc-shell PROC, append STRING."
  (when (buffer-live-p (process-buffer proc))
    (with-current-buffer (process-buffer proc)
      (let ((moving (= (point) (process-mark proc))))
        (save-excursion
          (let ((inhibit-read-only t))
            (goto-char (process-mark proc))
            (insert string)
            (set-marker (process-mark proc) (point))))
        (display-buffer (process-buffer proc))
        (if moving
            (with-selected-window (get-buffer-window (current-buffer))
              (goto-char (point-max))))))))

(defun govc-shell-run (name args buffer)
  "Run NAME command with ARGS in BUFFER."
  (with-current-buffer (if (stringp buffer) (get-buffer-create buffer) buffer)
    (let ((proc (get-buffer-process (current-buffer)))
          (process-environment (govc-environment))
          (session (govc-current-session))
          (inhibit-read-only t))
      (when proc
        (set-process-filter proc nil)
        (delete-process proc))
      (erase-buffer)
      (if (--any? (member (file-name-extension it) '("vmx" "vmdk")) args)
          (conf-mode)
        (govc-shell-mode))
      (govc-session-clone session)
      (setq-local govc-shell--revert-cmd `(govc-shell-run ,name ,args ,(current-buffer)))
      (setq mode-line-process '(:propertize ":run" face compilation-mode-line-run))
      (setq proc (apply 'start-process name (current-buffer) name args))
      (set-process-sentinel proc 'govc-shell-process-sentinel)
      (set-process-filter proc 'govc-shell-filter))))

(defun govc-shell-kill ()
  "Kill the process started by \\[govc-shell-command]."
  (interactive)
  (let ((buffer (current-buffer)))
    (if (get-buffer-process buffer)
        (interrupt-process (get-buffer-process buffer))
      (error "The %s process is not running" (downcase mode-name)))))

(defun govc-shell-process-sentinel (process event)
  "Process sentinel used by `govc-shell-run'.  When PROCESS exits EVENT is logged."
  (when (memq (process-status process) '(exit signal))
    (with-current-buffer (process-buffer process)
      (setq mode-line-process nil)
      (message "%s %s" (process-name process) (substring event 0 -1)))))

(defvar govc-shell-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-k") 'govc-shell-kill)
    map))

(define-derived-mode govc-shell-mode special-mode "govc-shell"
  "Mode for running govc commands."
  (setq-local font-lock-defaults '(govc-font-lock-keywords))
  (setq-local revert-buffer-function #'govc-shell--revert-function))

(defun govc-shell-command (&optional cmd buffer)
  "Shell CMD in BUFFER with current `govc-session' exported as GOVC_ env vars."
  (interactive)
  (let* ((session (govc-current-session))
         (args (if cmd (--map (format "%s" it) (-flatten (-non-nil (list govc-command cmd))))
                 (split-string-and-unquote (read-shell-command "command: " nil 'govc-command-history)))))
    (with-current-buffer (get-buffer-create (or buffer "*govc*"))
      (govc-session-clone session)
      (govc-shell-run (car args) (cdr args) (current-buffer)))))

(defcustom govc-max-events 100
  "Limit events output to the last N events."
  :type 'integer
  :group 'govc)

(defun govc-events ()
  "Events via govc events -n `govc-max-events'."
  (interactive)
  (govc-shell-command
   (list "events" "-l" "-n" govc-max-events (if current-prefix-arg "-f") (govc-selection)) "*govc-event*"))

(defun govc-tasks ()
  "Tasks via govc tasks."
  (interactive)
  (govc-shell-command
   (list "tasks" "-l" "-n" govc-max-events (if current-prefix-arg "-f") (govc-selection)) "*govc-task*"))

(defun govc-logs ()
  "Logs via govc logs -n `govc-max-events'."
  (interactive)
  (govc-shell-command
   (let ((host (govc-selection)))
     (list "logs" "-n" govc-max-events (if current-prefix-arg "-f") (if host (list "-host" host)))) "*govc-log*"))

(defun govc-parse-info (output)
  "Parse govc info command OUTPUT."
  (let* ((entries)
         (entry)
         (entry-key))
    (-each output
      (lambda (line)
        (let* ((ix (s-index-of ":" line))
               (key (s-trim (substring line 0 ix)))
               (val (s-trim (substring line (+ ix 1)))))
          (unless entry-key
            (setq entry-key key))
          (when (s-equals? key entry-key)
            (setq entry (make-hash-table :test 'equal))
            (add-to-list 'entries entry))
          (puthash key val entry))))
    entries))

(defun govc-table-column-names ()
  "Return a list of column names from `tabulated-list-format'."
  (--map (car (aref tabulated-list-format it))
         (number-sequence 0 (- (length tabulated-list-format) 1))))

(defun govc-table-column-value (key)
  "Return current column value for given KEY."
  (let ((names (govc-table-column-names))
        (entry (tabulated-list-get-entry))
        (value))
    (dotimes (ix (- (length names) 1))
      (if (s-equals? key (nth ix names))
          (setq value (elt entry ix))))
    value))

(defun govc-table-info (command &optional args)
  "Convert `govc-parse-info' COMMAND ARGS output to `tabulated-list-entries' format."
  (let ((names (govc-table-column-names)))
    (-map (lambda (info)
            (let ((id (or (gethash "Path" info)
                          (gethash (car names) info))))
              (list id (vconcat
                        (--map (or (gethash it info) "-")
                               names)))))
          (govc-parse-info (govc command args)))))

(defun govc-map-info (command &optional args)
  "Populate key=val map table with govc COMMAND ARGS output."
  (-map (lambda (line)
          (let* ((ix (s-index-of ":" line))
                 (key (s-trim (substring line 0 ix)))
                 (val (s-trim (substring line (+ ix 1)))))
            (list key (vector key val))))
        (govc command args)))

(defun govc-map-info-table (entries)
  "Tabulated `govc-map-info' data via ENTRIES."
  (let ((session (govc-current-session))
        (args (append govc-args (govc-selection)))
        (buffer (get-buffer-create "*govc-info*")))
    (pop-to-buffer buffer)
    (tabulated-list-mode)
    (setq govc-args args)
    (govc-session-clone session)
    (setq tabulated-list-format [("Name" 50)
                                 ("Value" 50)]
          tabulated-list-padding 2
          tabulated-list-entries entries)
    (tabulated-list-print)))

(defun govc-type-list-entries (command)
  "Convert govc COMMAND type table output to `tabulated-list-entries'."
  (-map (lambda (line)
          (let* ((entry (s-split-up-to " " (s-collapse-whitespace line) 2))
                 (name (car entry))
                 (type (nth 1 entry))
                 (value (car (last entry))))
            (list name (vector name type value))))
        (govc command govc-args)))

(defun govc-json-info-selection (command)
  "Run govc COMMAND -json on `govc-selection'."
  (if current-prefix-arg
      (--each (govc-selection) (govc-json-info command it))
    (govc-json-info command (govc-selection))))

(defun govc-json-diff ()
  "Diff two *govc-json* buffers in view."
  (let ((buffers))
    (-each (window-list-1)
      (lambda (w)
        (with-current-buffer (window-buffer w)
          (if (and (eq major-mode 'json-mode)
                   (s-starts-with? "*govc-json*" (buffer-name)))
              (push (current-buffer) buffers)))) )
    (if (= (length buffers) 2)
        (pop-to-buffer
         (diff-no-select (car buffers) (cadr buffers))))))

(defun govc-json-info (command selection)
  "Run govc COMMAND -json on SELECTION."
  (govc-process (govc-format-command command "-json" govc-args selection)
                (lambda ()
                  (let ((buffer (get-buffer-create (concat "*govc-json*" (if current-prefix-arg selection)))))
                    (copy-to-buffer buffer (point-min) (point-max))
                    (with-current-buffer buffer
                      (json-mode)
                      ;; We use `json-mode-beautify' as `json-pretty-print-buffer' does not work for `govc-host-json-info'
                      (json-mode-beautify))
                    (display-buffer buffer))))
  (if current-prefix-arg
      (govc-json-diff)))

(defun govc-mode-new-session ()
  "Connect new session for the current govc mode."
  (interactive)
  (call-interactively 'govc-session)
  (revert-buffer))

(defun govc-host-with-session ()
  "Host-mode with current session."
  (interactive)
  (govc-host nil (govc-current-session)))

(defun govc-vm-with-session ()
  "VM-mode with current session."
  (interactive)
  (govc-vm nil (govc-current-session)))

(defun govc-datastore-with-session ()
  "Datastore-mode with current session."
  (interactive)
  (govc-datastore nil (govc-current-session)))

(defun govc-pool-with-session ()
  "Pool-mode with current session."
  (interactive)
  (govc-pool nil (govc-current-session)))


;;; govc object mode
(defvar-local govc-object-history '("-")
  "History list of visited objects.")

(defun govc-object-collect ()
  "Wrapper for govc object.collect."
  (interactive)
  (let ((id (car govc-args)))
    (add-to-list 'govc-object-history id)
    (setq govc-session-path id))
  (govc-type-list-entries "object.collect"))

(defun govc-object-collect-selection (&optional json)
  "Expand object selection via govc object.collect.
Optionally specify JSON encoding."
  (interactive)
  (let* ((entry (or (tabulated-list-get-entry) (error "No entry")))
         (name (elt entry 0))
         (type (elt entry 1))
         (val (elt entry 2)))

    (setq govc-args (list (car govc-args) name))

    (cond
     ((s-blank? val))
     ((and (not json) (s-ends-with? "types.ManagedObjectReference" type))
      (let ((ids (govc "ls" "-L" (split-string val ","))))
        (setq govc-args (list (govc-object-prompt "moid: " ids)))))
     ((string= val "...")
      (if (s-starts-with? "[]" type) (setq json t))))

    (if json
        (govc-json-info "object.collect" nil)
      (tabulated-list-revert))))

(defun govc-object-collect-selection-json ()
  "JSON object selection via govc object.collect."
  (interactive)
  (govc-object-collect-selection t))

(defun govc-object-next ()
  "Next managed object reference."
  (interactive)
  (if (search-forward "types.ManagedObjectReference" nil t)
      (progn (govc-tabulated-list-unmark-all)
             (tabulated-list-put-tag (char-to-string dired-marker-char)))
    (goto-char (point-min))))

(defun govc-object-collect-parent ()
  "Parent object selection if reachable, otherwise prompt with `govc-object-history'."
  (interactive)
  (if (cadr govc-args)
      (let ((prop (butlast (split-string (cadr govc-args) "\\."))))
        (setq govc-args (list (car govc-args) (if prop (s-join "." prop)))))
    (save-excursion
      (goto-char (point-min))
      (if (re-search-forward "^[[:space:]]*parent" nil t)
          (govc-object-collect-selection)
        (let ((id (govc-object-prompt "moid: " govc-object-history)))
          (setq govc-args (list id (if (string= id "-") "content")))))))
  (tabulated-list-revert))

(defun govc-object (&optional moid property session)
  "Object browser aka MOB (Managed Object Browser).
Optionally starting at MOID and PROPERTY if given.
Inherit SESSION if given."
  (interactive)
  (let ((buffer (get-buffer-create "*govc-object*")))
    (if (called-interactively-p 'interactive)
        (switch-to-buffer buffer)
      (pop-to-buffer buffer))
    (govc-object-mode)
    (if session
        (govc-session-clone session)
      (call-interactively 'govc-session))
    (setq govc-args (list (or moid "-") property))
    (tabulated-list-print)))

(defun govc-object-info ()
  "Object browser via govc object.collect on `govc-selection'."
  (interactive)
  (if (equal major-mode 'govc-object-mode)
      (progn
        (setq govc-args (list (govc-object-prompt "moid: " govc-object-history)))
        (tabulated-list-revert))
    (govc-object (tabulated-list-get-id) nil (govc-current-session))))

(defvar govc-object-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "J" 'govc-object-collect-selection-json)
    (define-key map "N" 'govc-object-next)
    (define-key map "O" 'govc-object-info)
    (define-key map (kbd "DEL") 'govc-object-collect-parent)
    (define-key map (kbd "RET") 'govc-object-collect-selection)
    (define-key map "?" 'govc-object-popup)
    map)
  "Keymap for `govc-object-mode'.")

(define-derived-mode govc-object-mode govc-tabulated-list-mode "Object"
  "Major mode for handling a govc object."
  (setq tabulated-list-format [("Name" 40 t)
                               ("Type" 40 t)
                               ("Value" 40 t)]
        tabulated-list-padding 2
        tabulated-list-entries #'govc-object-collect)
  (tabulated-list-init-header))

(magit-define-popup govc-object-popup
  "Object popup."
  :actions (govc-keymap-popup govc-object-mode-map))


;;; govc metric mode
(defun govc-metric-sample ()
  "Sample metrics."
  (interactive)
  (govc-shell-command (list "metric.sample" govc-args govc-filter (govc-selection))))

(defun govc-metric-sample-plot ()
  "Plot metric sample."
  (interactive)
  (let* ((type (if (and (display-images-p) (not (eq current-prefix-arg '-))) 'png 'dumb))
         (max (if (member "-i" govc-args) "60" "180"))
         (args (append govc-args (list "-n" max "-plot" type govc-filter)))
         (session (govc-current-session))
         (metrics (govc-selection))
         (inhibit-read-only t))
    (with-current-buffer (get-buffer-create "*govc*")
      (govc-session-clone session)
      (erase-buffer)
      (delete-other-windows)
      (if (eq type 'dumb)
          (split-window-right)
        (split-window-below))
      (display-buffer-use-some-window (current-buffer) '((inhibit-same-window . t)))
      (--each metrics
        (let* ((cmd (govc-format-command "metric.sample" args it))
               (data (govc-process cmd 'buffer-string)))
          (if (eq type 'dumb)
              (insert data)
            (insert-image (create-image (string-as-unibyte data) type t))))))))

(defun govc-metric-select (metrics)
  "Select metric names.  METRICS is a regexp."
  (interactive (list (read-regexp "Select metrics" (regexp-quote ".usage."))))
  (save-excursion
    (goto-char (point-min))
    (while (not (eobp))
      (if (string-match-p metrics (tabulated-list-get-id))
          (govc-tabulated-list-mark)
        (govc-tabulated-list-unmark)))))

(defun govc-metric-info ()
  "Wrapper for govc metric.info."
  (govc-table-info "metric.info" (list govc-args (car govc-filter))))

(defvar govc-metric-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "RET") 'govc-metric-sample)
    (define-key map (kbd "P") 'govc-metric-sample-plot)
    (define-key map (kbd "s") 'govc-metric-select)
    map)
  "Keymap for `govc-metric-mode'.")

(defun govc-metric ()
  "Metrics info."
  (interactive)
  (let ((session (govc-current-session))
        (filter (or (govc-selection) (list govc-session-path)))
        (buffer (get-buffer-create "*govc-metric*")))
    (pop-to-buffer buffer)
    (govc-metric-mode)
    (govc-session-clone session)
    (if current-prefix-arg (setq govc-args '("-i" "300")))
    (setq govc-filter filter)
    (tabulated-list-print)))

(define-derived-mode govc-metric-mode govc-tabulated-list-mode "Metric"
  "Major mode for handling a govc metric."
  (setq tabulated-list-format [("Name" 35 t)
                               ("Group" 15 t)
                               ("Unit" 4 t)
                               ("Level" 5 t)
                               ("Summary" 50)]
        tabulated-list-sort-key (cons "Name" nil)
        tabulated-list-padding 2
        tabulated-list-entries #'govc-metric-info)
  (tabulated-list-init-header))


;;; govc host mode
(defun govc-ls-host ()
  "List hosts."
  (govc "ls" "-t" "HostSystem" "./..."))

(defun govc-esxcli-netstat-info ()
  "Wrapper for govc host.esxcli network ip connection list."
  (govc-table-info "host.esxcli"
                   (append govc-args '("-hints=false" "--" "network" "ip" "connection" "list"))))

(defun govc-esxcli-netstat (host)
  "Tabulated `govc-esxcli-netstat-info' HOST."
  (interactive (list (govc-object-prompt "Host: " 'govc-ls-host)))
  (let ((session (govc-current-session))
        (buffer (get-buffer-create "*govc-esxcli*")))
    (pop-to-buffer buffer)
    (tabulated-list-mode)
    (setq govc-args (list "-host" host))
    (govc-session-clone session)
    (setq tabulated-list-format [("CCAlgo" 10 t)
                                 ("ForeignAddress" 20 t)
                                 ("LocalAddress" 20 t)
                                 ("Proto" 5 t)
                                 ("RecvQ" 5 t)
                                 ("SendQ" 5 t)
                                 ("State" 15 t)
                                 ("WorldID" 7 t)
                                 ("WorldName" 10 t)]
          tabulated-list-padding 2
          tabulated-list-entries #'govc-esxcli-netstat-info)
    (tabulated-list-init-header)
    (tabulated-list-print)))

(defun govc-host-esxcli-netstat ()
  "Netstat via `govc-esxcli-netstat-info' with current host id."
  (interactive)
  (govc-esxcli-netstat (tabulated-list-get-id)))

(defun govc-host-info ()
  "Wrapper for govc host.info."
  (govc-table-info "host.info" (or govc-filter "*")))

(defun govc-host-json-info ()
  "JSON via govc host.info -json on current selection."
  (interactive)
  (govc-json-info-selection "host.info"))

(defvar govc-host-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "E" 'govc-events)
    (define-key map "L" 'govc-logs)
    (define-key map "J" 'govc-host-json-info)
    (define-key map "M" 'govc-metric)
    (define-key map "N" 'govc-host-esxcli-netstat)
    (define-key map "O" 'govc-object-info)
    (define-key map "T" 'govc-tasks)
    (define-key map "c" 'govc-mode-new-session)
    (define-key map "p" 'govc-pool-with-session)
    (define-key map "s" 'govc-datastore-with-session)
    (define-key map "v" 'govc-vm-with-session)
    (define-key map "?" 'govc-host-popup)
    map)
  "Keymap for `govc-host-mode'.")

(defun govc-host (&optional filter session)
  "Host info via govc.
Optionally filter by FILTER and inherit SESSION."
  (interactive)
  (let ((buffer (get-buffer-create "*govc-host*")))
    (pop-to-buffer buffer)
    (govc-host-mode)
    (if session
        (govc-session-clone session)
      (call-interactively 'govc-session))
    (setq govc-filter filter)
    (tabulated-list-print)))

(define-derived-mode govc-host-mode govc-tabulated-list-mode "Host"
  "Major mode for handling a list of govc hosts."
  (setq tabulated-list-format [("Name" 30 t)
                               ("Logical CPUs" 20 t)
                               ("CPU usage" 25 t)
                               ("Memory" 10 t)
                               ("Memory usage" 25 t)
                               ("Manufacturer" 13 t)
                               ("Boot time" 15 t)]
        tabulated-list-sort-key (cons "Name" nil)
        tabulated-list-padding 2
        tabulated-list-entries #'govc-host-info)
  (tabulated-list-init-header))

(magit-define-popup govc-host-popup
  "Host popup."
  :actions (govc-keymap-popup govc-host-mode-map))

(easy-menu-define govc-host-mode-menu govc-host-mode-map
  "Host menu."
  (cons "Host" (govc-keymap-menu govc-host-mode-map)))


;;; govc pool mode
(defun govc-pool-destroy (name)
  "Destroy pool with given NAME."
  (interactive (list (completing-read "Destroy pool: " (govc "ls" "-t" "ResourcePool" "host/*"))))
  (govc "pool.destroy" name))

(defun govc-pool-destroy-selection ()
  "Destroy via `govc-pool-destroy' on the pool selection."
  (interactive)
  (govc-do-selection 'govc-pool-destroy "Delete")
  (tabulated-list-revert))

(defun govc-pool-info ()
  "Wrapper for govc pool.info."
  (govc-table-info "pool.info" (list "-a" (or govc-filter (setq govc-filter "*")))))

(defun govc-pool-json-info ()
  "JSON via govc pool.info -json on current selection."
  (interactive)
  (govc-json-info-selection "pool.info"))

(defvar govc-pool-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "D" 'govc-pool-destroy-selection)
    (define-key map "E" 'govc-events)
    (define-key map "J" 'govc-pool-json-info)
    (define-key map "M" 'govc-metric)
    (define-key map "O" 'govc-object-info)
    (define-key map "T" 'govc-tasks)
    (define-key map "c" 'govc-mode-new-session)
    (define-key map "h" 'govc-host-with-session)
    (define-key map "s" 'govc-datastore-with-session)
    (define-key map "v" 'govc-vm-with-session)
    (define-key map "?" 'govc-pool-popup)
    map)
  "Keymap for `govc-pool-mode'.")

(defun govc-pool (&optional filter session)
  "Pool info via govc.
Optionally filter by FILTER and inherit SESSION."
  (interactive)
  (let ((buffer (get-buffer-create "*govc-pool*")))
    (pop-to-buffer buffer)
    (govc-pool-mode)
    (if session
        (govc-session-clone session)
      (call-interactively 'govc-session))
    (setq govc-filter filter)
    (tabulated-list-print)))

(define-derived-mode govc-pool-mode govc-tabulated-list-mode "Pool"
  "Major mode for handling a list of govc pools."
  (setq tabulated-list-format [("Name" 30 t)
                               ("CPU Usage" 25 t)
                               ("CPU Shares" 25 t)
                               ("CPU Reservation" 25 t)
                               ("CPU Limit" 10 t)
                               ("Mem Usage" 25 t)
                               ("Mem Shares" 25 t)
                               ("Mem Reservation" 25 t)
                               ("Mem Limit" 10 t)]
        tabulated-list-sort-key (cons "Name" nil)
        tabulated-list-padding 2
        tabulated-list-entries #'govc-pool-info)
  (tabulated-list-init-header))

(magit-define-popup govc-pool-popup
  "Pool popup."
  :actions (govc-keymap-popup govc-pool-mode-map))

(easy-menu-define govc-host-mode-menu govc-pool-mode-map
  "Pool menu."
  (cons "Pool" (govc-keymap-menu govc-pool-mode-map)))


;;; govc datastore mode
(defun govc-ls-datastore ()
  "List datastores."
  (govc "ls" "datastore"))

(defun govc-datastore-ls-entries ()
  "Wrapper for govc datastore.ls."
  (let* ((data (govc-json "datastore.ls" "-l" "-p" govc-filter))
         (file (plist-get (elt data 0) :File)))
    (-map (lambda (ent)
            (let ((name (plist-get ent :Path))
                  (size (plist-get ent :FileSize))
                  (time (plist-get ent :Modification))
                  (user (plist-get ent :Owner)))
              (list (concat govc-filter name)
                    (vector (file-size-human-readable size)
                            (current-time-string (date-to-time time))
                            name)))) file)))

(defun govc-datastore-ls-parent ()
  "Up to parent folder."
  (interactive)
  (if (s-blank? govc-filter)
      (let ((session (govc-current-session)))
        (govc-datastore-mode)
        (govc-session-clone session))
    (setq govc-filter (file-name-directory (directory-file-name govc-filter))))
  (tabulated-list-revert))

(defun govc-datastore-ls-child ()
  "Open datastore folder or file."
  (interactive)
  (let ((id (tabulated-list-get-id)))
    (if current-prefix-arg
        (govc-shell-command (list "datastore.ls" "-l" "-p" "-R" id))
      (if (s-ends-with? "/" id)
          (progn (setq govc-filter id)
                 (tabulated-list-revert))
        (govc-datastore-open)))))

(defun govc-datastore-open ()
  "Open datastore file."
  (lexical-let* ((srcfile (tabulated-list-get-id))
                 (srcpath (format "[%s] %s" (file-name-nondirectory govc-session-datastore) (s-chop-prefix "/" srcfile)))
                 (suffix (file-name-extension srcfile t))
                 (tmpfile (make-temp-file "govc-ds" nil suffix))
                 (session (govc-current-session)))
    (when (yes-or-no-p (concat "Open " srcpath "?"))
      (govc "datastore.download" srcfile tmpfile)
      (with-current-buffer (pop-to-buffer (find-file-noselect tmpfile))
        (govc-session-clone session)
        (add-hook 'kill-buffer-hook (lambda ()
                                      (with-demoted-errors
                                          (delete-file tmpfile))) t t)
        (add-hook 'after-save-hook (lambda ()
                                     (if (yes-or-no-p (concat "Upload changes to " srcpath "?"))
                                         (with-demoted-errors
                                             (govc "datastore.upload" tmpfile srcfile)))) t t)))))

(defun govc-datastore-tail (&optional file)
  "Tail datastore FILE."
  (interactive)
  (govc-shell-command
   (list "datastore.tail" "-n" govc-max-events (if current-prefix-arg "-f") (or file (govc-selection)))))

(defun govc-datastore-disk-info ()
  "Info datastore disk."
  (interactive)
  (delete-other-windows)
  (govc-shell-command
   (list "datastore.disk.info" "-uuid" (if current-prefix-arg "-c") (govc-selection))))

(defun govc-datastore-ls-json ()
  "JSON via govc datastore.ls -json on current selection."
  (interactive)
  (let ((govc-args '("-l" "-p")))
    (govc-json-info-selection "datastore.ls")))

(defun govc-datastore-ls-r-json ()
  "Search via govc datastore.ls -json -R on current selection."
  (interactive)
  (let ((govc-args '("-l" "-p" "-R")))
    (govc-json-info-selection "datastore.ls")))

(defun govc-datastore-mkdir (name)
  "Mkdir via govc datastore.mkdir with given NAME."
  (interactive (list (read-from-minibuffer "Create directory: " govc-filter)))
  (govc "datastore.mkdir" name)
  (tabulated-list-revert))

(defun govc-datastore-rm (paths)
  "Delete datastore PATHS."
  (--each paths (govc "datastore.rm" (if current-prefix-arg "-f") it)))

(defun govc-datastore-rm-selection ()
  "Delete selected datastore paths."
  (interactive)
  (govc-do-selection 'govc-datastore-rm "Delete")
  (tabulated-list-revert))

(defvar govc-datastore-ls-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "I" 'govc-datastore-disk-info)
    (define-key map "J" 'govc-datastore-ls-json)
    (define-key map "S" 'govc-datastore-ls-r-json)
    (define-key map "D" 'govc-datastore-rm-selection)
    (define-key map "T" 'govc-datastore-tail)
    (define-key map "+" 'govc-datastore-mkdir)
    (define-key map (kbd "DEL") 'govc-datastore-ls-parent)
    (define-key map (kbd "RET") 'govc-datastore-ls-child)
    (define-key map "?" 'govc-datastore-ls-popup)
    map)
  "Keymap for `govc-datastore-ls-mode'.")

(defun govc-datastore-ls (&optional datastore session filter)
  "List govc datastore.  Optionally specify DATASTORE, SESSION and FILTER."
  (interactive)
  (let ((buffer (get-buffer-create "*govc-datastore*")))
    (pop-to-buffer buffer)
    (govc-datastore-ls-mode)
    (if session
        (govc-session-clone session)
      (call-interactively 'govc-session))
    (setq govc-session-datastore (or datastore (govc-object-prompt "govc datastore: " 'govc-ls-datastore)))
    (setq govc-filter filter)
    (tabulated-list-print)))

(define-derived-mode govc-datastore-ls-mode govc-tabulated-list-mode "Datastore"
  "Major mode govc datastore.ls."
  (setq-local font-lock-defaults `(,(cdr govc-font-lock-keywords)))
  (setq tabulated-list-format [("Size" 10 t)
                               ("Modification time" 25 t)
                               ("Name" 40 t)]
        tabulated-list-sort-key (cons "Name" nil)
        tabulated-list-padding 2
        tabulated-list-entries #'govc-datastore-ls-entries)
  (tabulated-list-init-header))

(magit-define-popup govc-datastore-ls-popup
  "Datastore ls popup."
  :actions (govc-keymap-popup govc-datastore-ls-mode-map))

(easy-menu-define govc-datastore-ls-mode-menu govc-datastore-ls-mode-map
  "Datastore ls menu."
  (cons "Datastore" (govc-keymap-menu govc-datastore-ls-mode-map)))

(defvar govc-datastore-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "J" 'govc-datastore-json-info)
    (define-key map "M" 'govc-metric)
    (define-key map "O" 'govc-object-info)
    (define-key map (kbd "RET") 'govc-datastore-ls-selection)
    (define-key map "c" 'govc-mode-new-session)
    (define-key map "h" 'govc-host-with-session)
    (define-key map "p" 'govc-pool-with-session)
    (define-key map "v" 'govc-vm-with-session)
    (define-key map "?" 'govc-datastore-popup)
    map)
  "Keymap for `govc-datastore-mode'.")

(defun govc-datastore-json-info ()
  "JSON via govc datastore.info -json on current selection."
  (interactive)
  (govc-json-info-selection "datastore.info"))

(defun govc-datastore-info ()
  "Wrapper for govc datastore.info."
  (govc-table-info "datastore.info" (or govc-filter "*")))

(defun govc-datastore-ls-selection ()
  "Browse datastore."
  (interactive)
  (govc-datastore-ls (tabulated-list-get-id) (govc-current-session)))

(defun govc-datastore (&optional filter session)
  "Datastore info via govc.
Optionally filter by FILTER and inherit SESSION."
  (interactive)
  (let ((buffer (get-buffer-create "*govc-datastore*")))
    (pop-to-buffer buffer)
    (govc-datastore-mode)
    (if session
        (govc-session-clone session)
      (call-interactively 'govc-session))
    (setq govc-filter filter)
    (tabulated-list-print)
    (if (and govc-session-datastore (search-forward govc-session-datastore nil t))
        (beginning-of-line))))

(define-derived-mode govc-datastore-mode tabulated-list-mode "Datastore"
  "Major mode for govc datastore.info."
  (setq tabulated-list-format [("Name" 15 t)
                               ("Type" 10 t)
                               ("Capacity" 10 t)
                               ("Free" 10 t)
                               ("Remote" 30 t)]
        tabulated-list-sort-key (cons "Name" nil)
        tabulated-list-padding 2
        tabulated-list-entries #'govc-datastore-info)
  (tabulated-list-init-header))

(magit-define-popup govc-datastore-popup
  "Datastore popup."
  :actions (govc-keymap-popup govc-datastore-mode-map))

(easy-menu-define govc-datastore-mode-menu govc-datastore-mode-map
  "Datastore menu."
  (cons "Datastore" (govc-keymap-menu govc-datastore-mode-map)))


;;; govc vm mode
(defun govc-vm-prompt (prompt)
  "PROMPT for a vm name."
  (completing-read prompt (govc "ls" "vm")))

(defun govc-vm-start (name)
  "Start vm with given NAME."
  (interactive (list (govc-vm-prompt "Start vm: ")))
  (govc "vm.power" "-on" name))

(defun govc-vm-shutdown (name)
  "Shutdown vm with given NAME."
  (interactive (list (govc-vm-prompt "Shutdown vm: ")))
  (govc "vm.power" "-s" "-force" name))

(defun govc-vm-reboot (name)
  "Reboot vm with given NAME."
  (interactive (list (govc-vm-prompt "Reboot vm: ")))
  (govc "vm.power" "-r" "-force" name))

(defun govc-vm-suspend (name)
  "Suspend vm with given NAME."
  (interactive (list (govc-vm-prompt "Suspend vm: ")))
  (govc "vm.power" "-suspend" name))

(defun govc-vm-destroy (name)
  "Destroy vm with given NAME."
  (interactive (list (govc-vm-prompt "Destroy vm: ")))
  (govc "vm.destroy" name))

(defun govc-vm-vnc-enable (name)
  "Enable vnc on vm with given NAME."
  (--map (last (split-string it))
         (govc "vm.vnc" "-enable"
               "-port" "-1"
               "-password" (format "%08x" (random (expt 16 8))) name)))

(defun govc-vm-vnc (name &optional arg)
  "VNC for vm with given NAME.
By default, enable and open VNC for the given vm NAME.
With prefix \\[negative-argument] ARG, VNC will be disabled.
With prefix \\[universal-argument] ARG, VNC will be enabled but not opened."
  (interactive (list (govc-vm-prompt "VNC vm: ")
                     current-prefix-arg))
  (if (equal arg '-)
      (govc "vm.vnc" "-disable" name)
    (let ((urls (govc-vm-vnc-enable name)))
      (unless arg
        (-each (-flatten urls) 'browse-url)))))

(defun govc-vm-console (name &optional arg)
  "Console for vm with given NAME.
By default, displays a console screen capture.
With prefix \\[universal-argument] ARG, launches an interactive console (VMRC)."
  (interactive (list (govc-vm-prompt "Console vm: ")
                     current-prefix-arg))
  (if arg
      (browse-url (car (govc "vm.console" name)))
    (let* ((data (govc-process (govc-format-command "vm.console" "-capture" "-" name) 'buffer-string))
           (inhibit-read-only t))
      (with-current-buffer (get-buffer-create "*govc*")
        (erase-buffer)
        (insert-image (create-image (string-as-unibyte data) 'png t))
        (read-only-mode)
        (display-buffer (current-buffer))))))

(defun govc-vm-start-selection ()
  "Start via `govc-vm-start' on the current selection."
  (interactive)
  (govc-vm-start (govc-selection))
  (tabulated-list-revert))

(defun govc-vm-shutdown-selection ()
  "Shutdown via `govc-vm-shutdown' on the current selection."
  (interactive)
  (govc-vm-shutdown (govc-selection))
  (tabulated-list-revert))

(defun govc-vm-reboot-selection ()
  "Reboot via `govc-vm-reboot' on the current selection."
  (interactive)
  (govc-vm-reboot (govc-selection))
  (tabulated-list-revert))

(defun govc-vm-suspend-selection ()
  "Suspend via `govc-vm-suspend' on the current selection."
  (interactive)
  (govc-vm-suspend (govc-selection))
  (tabulated-list-revert))

(defun govc-vm-destroy-selection ()
  "Destroy via `govc-vm-destroy' on the current selection."
  (interactive)
  (govc-do-selection 'govc-vm-destroy "Destroy")
  (tabulated-list-revert))

(defun govc-vm-vnc-selection ()
  "VNC via `govc-vm-vnc' on the current selection."
  (interactive)
  (govc-vm-vnc (govc-selection) current-prefix-arg))

(defun govc-vm-console-selection ()
  "Console via `govc-vm-console' on the current selection."
  (interactive)
  (govc-vm-console (tabulated-list-get-id) current-prefix-arg))

(defun govc-vm-info ()
  "Wrapper for govc vm.info."
  (unless (string-empty-p govc-session-datacenter)
    (govc-table-info "vm.info" (list "-r" (or govc-filter (setq govc-filter "*"))))))

(defun govc-vm-host ()
  "Host info via `govc-host' with host(s) of current selection."
  (interactive)
  (govc-host (concat "*/" (govc-table-column-value "Host"))
             (govc-current-session)))

(defun govc-vm-log-directory ()
  "VM log directory of current selection."
  (car (govc "object.collect" "-s" (tabulated-list-get-id) "config.files.logDirectory")))

(defun govc-vm-datastore ()
  "Datastore via `govc-datastore-ls' with datastore of current selection."
  (interactive)
  (if current-prefix-arg
      (govc-datastore (s-split ", " (govc-table-column-value "Storage") t)
                      (govc-current-session))
    (let* ((dir (govc-vm-log-directory))
           (args (s-split "\\[\\|\\]" dir t)))
      (govc-datastore-ls (first args) (govc-current-session) (concat (s-trim (second args)) "/")))))

(defun govc-vm-logs ()
  "Logs via `govc-datastore-tail' with logDirectory of current selection."
  (interactive)
  (if (tabulated-list-get-id)
      (govc-datastore-tail (concat (govc-vm-log-directory) "/vmware.log"))
    (govc-logs)))

(defun govc-vm-ping ()
  "Ping VM."
  (interactive)
  (let ((ping-program-options '("-c" "20")))
    (ping (govc-table-column-value "IP address"))))

(defun govc-vm-device-ls ()
  "Devices via `govc-device' on the current selection."
  (interactive)
  (govc-device (tabulated-list-get-id)
               (govc-current-session)))

(defun govc-vm-extra-config ()
  "Populate table with govc vm.info -e output."
  (let* ((data (govc-json "vm.info" govc-args))
         (vms (plist-get data :VirtualMachines))
         (info))
    (mapc
     (lambda (vm)
       (let* ((config (plist-get vm :Config))
              (name (plist-get config :Name)))
         (mapc (lambda (x)
                 (let ((key (plist-get x :Key))
                       (val (plist-get x :Value)))
                   (push (list key (vector key val)) info)))
               (plist-get config :ExtraConfig))
         (if (> (length vms) 1)
             (push (list name (vector "vm.name" name)) info))))
     vms)
    info))

(defun govc-vm-extra-config-table ()
  "ExtraConfig via `govc-vm-extra-config' on the current selection."
  (interactive)
  (govc-map-info-table #'govc-vm-extra-config))

(defun govc-vm-json-info ()
  "JSON via govc vm.info -json on current selection."
  (interactive)
  (govc-json-info-selection "vm.info"))

(defvar govc-vm-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "E" 'govc-events)
    (define-key map "L" 'govc-vm-logs)
    (define-key map "J" 'govc-vm-json-info)
    (define-key map "O" 'govc-object-info)
    (define-key map "T" 'govc-tasks)
    (define-key map "X" 'govc-vm-extra-config-table)
    (define-key map (kbd "RET") 'govc-vm-device-ls)
    (define-key map "C" 'govc-vm-console-selection)
    (define-key map "V" 'govc-vm-vnc-selection)
    (define-key map "D" 'govc-vm-destroy-selection)
    (define-key map "^" 'govc-vm-start-selection)
    (define-key map "!" 'govc-vm-shutdown-selection)
    (define-key map "@" 'govc-vm-reboot-selection)
    (define-key map "&" 'govc-vm-suspend-selection)
    (define-key map "H" 'govc-vm-host)
    (define-key map "M" 'govc-metric)
    (define-key map "P" 'govc-vm-ping)
    (define-key map "S" 'govc-vm-datastore)
    (define-key map "c" 'govc-mode-new-session)
    (define-key map "h" 'govc-host-with-session)
    (define-key map "p" 'govc-pool-with-session)
    (define-key map "s" 'govc-datastore-with-session)
    (define-key map "?" 'govc-vm-popup)
    map)
  "Keymap for `govc-vm-mode'.")

(defun govc-vm (&optional filter session)
  "VM info via govc.
Optionally filter by FILTER and inherit SESSION."
  (interactive)
  (let ((buffer (get-buffer-create "*govc-vm*")))
    (pop-to-buffer buffer)
    (govc-vm-mode)
    (if session
        (govc-session-clone session)
      (call-interactively 'govc-session))
    (setq govc-filter filter)
    (tabulated-list-print)))

(define-derived-mode govc-vm-mode govc-tabulated-list-mode "VM"
  "Major mode for handling a list of govc vms."
  (setq tabulated-list-format [("Name" 40 t)
                               ("Power state" 12 t)
                               ("Boot time" 13 t)
                               ("IP address" 15 t)
                               ("Guest name" 20 t)
                               ("Host" 20 t)
                               ("CPU usage" 15 t)
                               ("Host memory usage" 18 t)
                               ("Guest memory usage" 19 t)
                               ("Storage committed" 18 t)
                               ("Storage" 10 t)
                               ("Network" 10 t)]
        tabulated-list-sort-key (cons "Name" nil)
        tabulated-list-padding 2
        tabulated-list-entries #'govc-vm-info)
  (tabulated-list-init-header))

(magit-define-popup govc-vm-popup
  "VM popup."
  :actions (govc-keymap-popup govc-vm-mode-map))

(easy-menu-define govc-vm-mode-menu govc-vm-mode-map
  "VM menu."
  (cons "VM" (govc-keymap-menu govc-vm-mode-map)))


;;; govc device mode
(defun govc-device-ls ()
  "Wrapper for govc device.ls -vm VM."
  (govc-type-list-entries "device.ls"))

(defun govc-device-info ()
  "Populate table with govc device.info output."
  (govc-map-info "device.info" govc-args))

(defun govc-device-info-table ()
  "Tabulated govc device.info."
  (interactive)
  (govc-map-info-table #'govc-device-info))

(defun govc-device-json-info ()
  "JSON via govc device.info -json on current selection."
  (interactive)
  (govc-json-info-selection "device.info"))

(defvar govc-device-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "J") 'govc-device-json-info)
    (define-key map (kbd "RET") 'govc-device-info-table)
    map)
  "Keymap for `govc-device-mode'.")

(defun govc-device (&optional vm session)
  "List govc devices for VM.  Optionally inherit SESSION."
  (interactive)
  (let ((buffer (get-buffer-create "*govc-device*")))
    (pop-to-buffer buffer)
    (govc-device-mode)
    (if session
        (govc-session-clone session)
      (call-interactively 'govc-session))
    (setq govc-args (list "-vm" (or vm (govc-vm-prompt "vm: "))))
    (tabulated-list-print)))

(define-derived-mode govc-device-mode govc-tabulated-list-mode "Device"
  "Major mode for handling a govc device."
  (setq tabulated-list-format [("Name" 15 t)
                               ("Type" 30 t)
                               ("Summary" 40 t)]
        tabulated-list-sort-key (cons "Name" nil)
        tabulated-list-padding 2
        tabulated-list-entries #'govc-device-ls)
  (tabulated-list-init-header))

(magit-define-popup govc-popup
  "govc popup."
  :actions (govc-keymap-list govc-command-map))

(easy-menu-change
 '("Tools") "govc"
 (govc-keymap-menu govc-command-map)
 "Search Files (Grep)...")

(provide 'govc)

;;; govc.el ends here
