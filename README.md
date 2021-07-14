# Escape room1

A grid-world environment with non-Markovian observations.

## Install
This package can be installed as usual:

    pip install .

Or, we can install a specific tested version of this package and its dependencies with:

    poetry install --no-dev

Omit the `--no-dev` option if you're installing for local development.

## Use

Import the package use the gym environment:

    from escape_room1 import EscapeRoom1
		env = EscapeRoom1()

The environment is also registered under the name "EscapeRoom1-v0" for `gym.make`.
