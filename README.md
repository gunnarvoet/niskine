# niskine
NISKINe mooring data analysis.

## Python Computing Environment
The python environment is defined in [environment.yml](environment.yml). The environment can be generated via
`conda env create -f environment.yml` and then activated via `conda activate niskine`.

The local [niskine](niskine) package is automatically installed into the environment and can simply be imported as
```python
import niskine
```

## Data Directory
The data directory has been excluded via [.gitignore](.gitignore) to avoid adding large files to the repository. The structure of the data directory, however, is defined via [config.yml](config.yml) and can be generated locally.

For example, processed (netcdf) ADCP data files are stored under `data/proc/adcp/`. Copy or link the files into this directory. Running
```python
niskine.io.link_proc_adcp(<mooring_data_dir>)
```
will link the files if `<mooring_data_dir>` has the same structure as the mooring directory on kipapa.

## Collaborating on GitHub
Fork the repository from https://github.com/modscripps/niskine to your GitHub account and clone the repository to your local computer from there.

Add the primary repository:
```sh
git remote add upstream https://github.com/modscripps/niskine.git
```

To make changes, create a new branch via
```
git checkout -b <branch-name>
```
and hack away. Add and commit your changes using `git add` and `git commit`.
Push the branch to your personal GitHub account with
```
git push -u origin <branch-name>
```
When done developing, head to your fork on GitHub and click `Compare & pull request` to merge the changes onto the main branch of the primary repository.
Pull the changes onto the main branch on your local computer via
```
git fetch upstream && git merge upstream/main
```
You may then delete your local branch via
```
git branch -d <branch-name>
```
and also push the changes on the main branch to your personal GitHub repository with a simple `git push`. Alternatively, the personal GitHub repository also has a button to fetch & merge from the primary repository.
