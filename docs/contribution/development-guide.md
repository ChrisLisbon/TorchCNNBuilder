# Development

We try to maintain good practices of readable open source code. 
Therefore, if you want to participate in the development and open your pool request, pay attention to the following points:
- Every push is checked by the flake8 job. It will show you PEP8 errors or possible code improvements.
- Use this linter script after your code:

```bash
bash lint_and_check.sh
```

*You can mark function docstrings using `#noqa`, in order for flake8 not to pay attention to them.*

# General tips

- If it's possible, try to create pull-requests by using fork
- Give only appropriate names to commits / issues / pull-requests
- It's better to use `pyenv`, `conda` or some different options of python environments in order to develop

# Release process

Despite the fact that the framework is very small, I want to maintain its consistency. 
The release procedure looks like this:

- pull-request is approved by maintainers and merged with squashing commits
- a new tag is being released to the github repository
- a new tag is being released in pypi

