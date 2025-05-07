#!/usr/bin/env -S just --justfile

# Cross platform shebang:
shebang := if os() == 'windows' {
  'pwsh.exe'
} else {
  '/usr/bin/env bash'
}

set shell := ["/usr/bin/env", "pwsh" ,"-noprofile", "-nologo", "-c"]
# set shell := ["/usr/bin/env", "bash" ,"-c"]
set windows-shell := ["pwsh.exe","-NoLogo", "-noprofile", "-c"]

