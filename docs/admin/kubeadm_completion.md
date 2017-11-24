
Output shell completion code for the specified shell (bash or zsh).

### Synopsis



Output shell completion code for the specified shell (bash or zsh).
The shell code must be evalutated to provide interactive
completion of kubeadm commands. This can be done by sourcing it from
the .bash_profile.

Note: this requires the bash-completion framework, which is not installed
by default on Mac. This can be installed by using homebrew:

    $ brew install bash-completion

Once installed, bash_completion must be evaluated. This can be done by adding the
following line to the .bash_profile

    $ source $(brew --prefix)/etc/bash_completion

Note for zsh users: [1] zsh completions are only supported in versions of zsh >= 5.2

```
kubeadm completion SHELL
```

### Examples

```

# Install bash completion on a Mac using homebrew
brew install bash-completion
printf "\n# Bash completion support\nsource $(brew --prefix)/etc/bash_completion\n" >> $HOME/.bash_profile
source $HOME/.bash_profile

# Load the kubeadm completion code for bash into the current shell
source <(kubeadm completion bash)

# Write bash completion code to a file and source if from .bash_profile
kubeadm completion bash > ~/.kube/kubeadm_completion.bash.inc
printf "\n# Kubeadm shell completion\nsource '$HOME/.kube/kubeadm_completion.bash.inc'\n" >> $HOME/.bash_profile
source $HOME/.bash_profile

# Load the kubeadm completion code for zsh[1] into the current shell
source <(kubeadm completion zsh)
```

