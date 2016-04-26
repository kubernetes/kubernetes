;;; govc.el --- Interface to govc for managing VMware ESXi and vCenter

;; Author: The govc developers
;; URL: https://github.com/vmware/govmomi/tree/master/govc/emacs
;; Keywords: convenience
;; Version: 0.1.0
;; Package-Requires: ((emacs "24.3") (dash "1.5.0") (s "1.9.0") (magit-popup "2.0.50") (json-mode "1.6.0"))

;; This file is NOT part of GNU Emacs.

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 3, or (at your option)
;; any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with GNU Emacs; see the file COPYING.  If not, write to the
;; Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
;; Boston, MA 02110-1301, USA.

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
(require 'dired)
(require 'json-mode)
(require 'magit-popup)
(require 'url-parse)
(require 's)

(defgroup govc nil
  "Emacs customization group for govc."
  :group 'convenience)

(defcustom govc-keymap-prefix "C-c ;"
  "Prefix for `govc-mode'."
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
    (kill-new (message "%s" (s-join " " (--map (format "'%s'" it) (govc-selection)))))))

(defvar govc-font-lock-keywords
  (list
   (list dired-re-mark '(0 dired-mark-face))))

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
(defvar govc-urls nil
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

When in `govc-vm' or `govc-host' mode, a default URL is composed with the
current session credentials and the IP address of the current vm/host and
the vm/host name as the session name.  This makes it easier to connect to
nested ESX/vCenter VMs or directly to an ESX host.")

(defvar-local govc-session-url nil
  "ESX or vCenter URL set by `govc-session' via `govc-urls' selection.")

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
        (or (url-target url) (url-host url)))))

(defun govc-command (command &rest args)
  "Format govc COMMAND ARGS."
  (format "govc %s %s" command
          (s-join " " (--map (format "'%s'" it)
                             (-flatten (-non-nil args))))))

(defconst govc-environment-map (--map (cons (concat "GOVC_" (upcase it))
                                            (intern (concat "govc-session-" it)))
                                      '("url" "insecure" "datacenter" "datastore"))

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
  (kill-new (message (if arg (s-join " " (govc-export-environment arg)) govc-session-url))))

(defun govc-process (command handler)
  "Run COMMAND, calling HANDLER upon successful exit of the process."
  (message command)
  (let ((process-environment (govc-environment))
        (exit-code))
    (with-temp-buffer
      (setq exit-code (call-process-shell-command command nil (current-buffer)))
      (if (zerop exit-code)
          (funcall handler)
        (error (buffer-string))))))

(defun govc (command &rest args)
  "Execute govc COMMAND with ARGS.
Return value is `buffer-string' split on newlines."
  (govc-process (govc-command command args)
                (lambda ()
                  (split-string (buffer-string) "\n" t))))

(defun govc-json (command &rest args)
  "Execute govc COMMAND passing arguments ARGS.
Return value is `json-read'."
  (govc-process (govc-command command (cons "-json" args))
                (lambda ()
                  (goto-char (point-min))
                  (let ((json-object-type 'plist))
                    (json-read)))))

(defun govc-ls-datacenter ()
  "List datacenters."
  (delete-dups (--map (nth 1 (split-string it "/"))
                      (govc "ls"))))

(defun govc-object-prompt (prompt ls)
  "PROMPT for object name via LS function.  Return object without PROMPT if there is just one instance."
  (let ((objs (funcall ls)))
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
                 (setf (url-target url) (govc-table-column-value "Name"))))
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

(defun govc-session-set-url (url)
  "Set `govc-session-url' to URL and optionally set other govc-session-* variables via URL query."
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

(defun govc-shell-command (&optional cmd)
  "Shell CMD with current `govc-session' exported as GOVC_ env vars."
  (interactive)
  (let ((process-environment (govc-environment))
        (current-prefix-arg "*govc*")
        (url govc-session-url))
    (if cmd
        (async-shell-command cmd current-prefix-arg)
      (call-interactively 'async-shell-command))
    (with-current-buffer (get-buffer current-prefix-arg)
      (setq govc-session-url url))))

(defcustom govc-max-events 50
  "Limit events output to the last N events."
  :type 'integer
  :group 'govc)

(defun govc-events ()
  "Events via govc events -n `govc-max-events'."
  (interactive)
  (govc-shell-command
   (govc-command "events"
                 (list "-n" govc-max-events (govc-selection)))))

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

(defun govc-json-info (command &optional selection)
  "Run govc COMMAND -json on SELECTION."
  (interactive)
  (govc-process (govc-command command (append (cons "-json" govc-args)
                                              (or selection (govc-selection))))
                (lambda ()
                  (let ((buffer (get-buffer-create "*govc-json*")))
                    (with-current-buffer buffer
                      (erase-buffer))
                    (copy-to-buffer buffer (point-min) (point-max))
                    (pop-to-buffer buffer)
                    (json-mode)
                    ;; We use `json-mode-beautify' as `json-pretty-print-buffer' does not work for `govc-host-json-info'
                    (json-mode-beautify)
                    (goto-char (point-min))))))

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


;;; govc host mode
(defun govc-ls-host ()
  "List hosts."
  (govc "ls" "-t" "HostSystem" "host/*"))

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
    (setq govc-args (list "-host.ipath" host))
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
  (govc-table-info "host.info" (or govc-filter "*/*")))

(defun govc-host-json-info ()
  "JSON via govc host.info -json on current selection."
  (interactive)
  (govc-json-info "host.info" (govc-selection)))

(defvar govc-host-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "E" 'govc-events)
    (define-key map "J" 'govc-host-json-info)
    (define-key map "N" 'govc-host-esxcli-netstat)
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
(defun govc-ls-pool (&optional pools)
  "List resource POOLS recursively."
  (let ((subpools (govc "ls" "-t" "ResourcePool" (--map (concat it "/*") (or pools '("host"))))))
    (append pools
            (if subpools
                (govc-ls-pool subpools)))))

(defun govc-ls-vapp ()
  "List virtual apps."
  (govc "ls" "-t" "VirtualApp" "vm"))

(defun govc-pool-destroy (name)
  "Destroy pool with given NAME."
  (interactive (list (completing-read "Destroy pool: " (govc-ls-pool))))
  (govc "pool.destroy" name))

(defun govc-pool-destroy-selection ()
  "Destroy via `govc-pool-destroy' on the pool selection."
  (interactive)
  (govc-do-selection 'govc-pool-destroy "Delete")
  (tabulated-list-revert))

(defun govc-pool-info ()
  "Wrapper for govc pool.info."
  (govc-table-info "pool.info" (or govc-filter (append (govc-ls-pool) (govc-ls-vapp)))))

(defun govc-pool-json-info ()
  "JSON via govc pool.info -json on current selection."
  (interactive)
  (govc-json-info "pool.info" (govc-selection)))

(defvar govc-pool-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "E" 'govc-events)
    (define-key map "J" 'govc-pool-json-info)
    (define-key map "D" 'govc-pool-destroy-selection)
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
    (if (s-ends-with? "/" id)
        (progn (setq govc-filter id)
               (tabulated-list-revert))
      (govc-datastore-open))))

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

(defun govc-datastore-ls-json ()
  "JSON via govc datastore.ls -json on current selection."
  (interactive)
  (let ((govc-args '("-l" "-p")))
    (govc-json-info "datastore.ls" (govc-selection))))

(defun govc-datastore-mkdir (name)
  "Mkdir via govc datastore.mkdir with given NAME."
  (interactive (list (read-from-minibuffer "Create directory: " govc-filter)))
  (govc "datastore.mkdir" name)
  (tabulated-list-revert))

(defun govc-datastore-rm (paths)
  "Delete datastore PATHS."
  (--each paths (govc "datastore.rm" it)))

(defun govc-datastore-rm-selection ()
  "Delete selected datastore paths."
  (interactive)
  (govc-do-selection 'govc-datastore-rm "Delete")
  (tabulated-list-revert))

(defvar govc-datastore-ls-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "J" 'govc-datastore-ls-json)
    (define-key map "D" 'govc-datastore-rm-selection)
    (define-key map "+" 'govc-datastore-mkdir)
    (define-key map (kbd "DEL") 'govc-datastore-ls-parent)
    (define-key map (kbd "RET") 'govc-datastore-ls-child)
    (define-key map "?" 'govc-datastore-ls-popup)
    map)
  "Keymap for `govc-datastore-ls-mode'.")

(defun govc-datastore-ls (&optional datastore session)
  "List govc datastore.  Optionally specify DATASTORE and SESSION."
  (interactive)
  (let ((buffer (get-buffer-create "*govc-datastore*")))
    (pop-to-buffer buffer)
    (govc-datastore-ls-mode)
    (if session
        (govc-session-clone session)
      (call-interactively 'govc-session))
    (setq govc-session-datastore (or datastore (govc-object-prompt "govc datastore: " 'govc-ls-datastore)))
    (tabulated-list-print)))

(define-derived-mode govc-datastore-ls-mode govc-tabulated-list-mode "Datastore"
  "Major mode govc datastore.ls."
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
  (govc-json-info "datastore.info"))

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
    (tabulated-list-print)))

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

(defun govc-vm-screen (name &optional arg)
  "Console screenshot of vm NAME console.
Open via `eww' by default, via `browse-url' if ARG is non-nil."
  (interactive (list (govc-vm-prompt "Console screenshot vm: ")
                     current-prefix-arg))
  (let* ((data (govc-json "vm.info" name))
         (vms (plist-get data :VirtualMachines))
         (url (govc-url-parse govc-session-url)))
    (mapc
     (lambda (vm)
       (let* ((moid (plist-get (plist-get vm :Self) :Value))
              (on (string= "poweredOn" (plist-get (plist-get vm :Runtime) :PowerState)))
              (host (format "%s:%d" (url-host url) (or (url-port url) 443)))
              (path (concat "/screen?id=" moid))
              (auth (concat (url-user url) ":" (url-password url))))
         (if current-prefix-arg
             (browse-url (concat "https://" auth "@" host path))
           (let ((creds `((,host ("VMware HTTP server" . ,(base64-encode-string auth)))))
                 (url-basic-auth-storage 'creds)
                 (u (concat "https://" host path)))
             (require 'eww)
             (if on
                 (url-retrieve u 'eww-render (list u))
               (kill-new (message u)))))))
     vms)))

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

(defun govc-vm-screen-selection ()
  "Console screenshot via `govc-vm-screen' on the current selection."
  (interactive)
  (govc-vm-screen (govc-selection) current-prefix-arg))

(defun govc-vm-info ()
  "Wrapper for govc vm.info."
  (govc-table-info "vm.info" (list "-r" (or govc-filter (setq govc-filter (govc-vm-filter))))))

(defun govc-vm-host ()
  "Host info via `govc-host' with host(s) of current selection."
  (interactive)
  (govc-host (concat "*/" (govc-table-column-value "Host"))
             (govc-current-session)))

(defun govc-vm-datastore ()
  "Datastore via `govc-datastore-ls' with datastore of current selection."
  (interactive)
  (govc-datastore (s-split ", " (govc-table-column-value "Storage") t)
                  (govc-current-session)))

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
  (govc-json-info "vm.info"))

(defvar govc-vm-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map "E" 'govc-events)
    (define-key map "J" 'govc-vm-json-info)
    (define-key map "X" 'govc-vm-extra-config-table)
    (define-key map (kbd "RET") 'govc-vm-device-ls)
    (define-key map "C" 'govc-vm-screen-selection)
    (define-key map "V" 'govc-vm-vnc-selection)
    (define-key map "D" 'govc-vm-destroy-selection)
    (define-key map "^" 'govc-vm-start-selection)
    (define-key map "!" 'govc-vm-shutdown-selection)
    (define-key map "@" 'govc-vm-reboot-selection)
    (define-key map "&" 'govc-vm-suspend-selection)
    (define-key map "H" 'govc-vm-host)
    (define-key map "S" 'govc-vm-datastore)
    (define-key map "P" 'govc-vm-ping)
    (define-key map "c" 'govc-mode-new-session)
    (define-key map "h" 'govc-host-with-session)
    (define-key map "p" 'govc-pool-with-session)
    (define-key map "s" 'govc-datastore-with-session)
    (define-key map "?" 'govc-vm-popup)
    map)
  "Keymap for `govc-vm-mode'.")

(defun govc-vm-filter ()
  "Default `govc-filter' for `vm-info'."
  (--map (concat it "/*")
         (append (govc-ls-folder (list (concat "/" govc-session-datacenter "/vm")))
                 (govc "ls" "-t" "VirtualApp" "vm"))))

(defun govc-ls-folder (folders)
  "List FOLDERS recursively."
  (let ((subfolders (govc "ls" "-t" "Folder" folders)))
    (append folders
            (if subfolders
                (govc-ls-folder subfolders)))))

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
  (-map (lambda (line)
          (let* ((entry (s-split-up-to " " (s-collapse-whitespace line) 2))
                 (name (car entry))
                 (type (nth 1 entry))
                 (summary (car (last entry))))
            (list name (vector name type summary))))
        (govc "device.ls" govc-args)))

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
  (govc-json-info "device.info"))

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
