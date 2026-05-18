# 👩‍🏫 CLimaVAR API
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📖 About
ClimaVAR is an AI-powered tool launching ahead of COP30 that helps anyone spot and understand climate misinformation. Like VAR in football helps referees to review decisions, ClimaVAR reviews questionable climate claims - giving communities the tools to fight back against false narratives that delay action and damage what we love.

**Helpful links**: 
* [ClimaVAR website](https://climavar.com/)

## 👨‍💻 Branching style
Branches should be created for every 'Task' or 'Feature'. Branches can also be created for hot fixes or infrastructure. Branch from master with the following format:
    ```
    # Features (Tasks)
    feat/small-feature-description

    # Infrastructure
    infr/small-step-message

    # Issues and Bugs
    hotfix/small-task-message
    ```

2. Before pushing, run the tests and lint your code with `black`. The CI will shout 
   at you otherwise.

3. When making a pull request, follow the pull request template provided in the 
   description.

4. After merging a pull request into master, please delete the branch to avoid clutter.

## Setup


### MacOS
#### Back end
1. Navigate to the root of your cloned repository.
2. From the terminal, in the root of the cloned repository, run ``cd backend`` to navigate to the ``/backend/`` folder.
3. Run ``python3 -m venv .venv`` to create a virtual environment (.venv).
4. Run ``. .venv/bin/activate`` to activate your virtual environment. You should see your terminal prefixed by ``.venv`` if this has been successful.
5. Run ``export APP_DEVELOPMENT=True`` to set ``APP_DEVELOPMENT`` environment variable to ``True``. This will bypass the need for a secret key and allow you to develop.
6. Run ``pip install -r requirements.txt`` to install project back end dependencies.
7. Run ``python3 manage.py runserver`` to run the server.
8. You will likely have migrations (i.e., ``You have N unapplied migration(s).``). To fix this, simply run ``python3 manage.py migrate``. 
9. To create a superuser, run ``python3 manage.py createsuperuser``. Provide the username, email, and password as prompted by your terminal. This will be the account you use to access the admin panel.


### Windows
#### Back end
1. Navigate to the root of your cloned repository.
2. From the terminal, in the root of the cloned repository, run ``cd backend`` to navigate to the ``/backend/`` folder.
3. Run ``python -m venv .venv`` to create a virtual environment (.venv).
4. Run ``.venv/Scripts/activate`` or ``.venv/Scripts/activate.bat`` to activate your virtual environment. You should see your terminal prefixed by ``.venv`` if this has been successful.
5. Run ``$env:APP_DEVELOPMENT="true"`` to set ``APP_DEVELOPMENT`` environment variable to ``True``. This will bypass the need for a secret key and allow you to develop.
6. Run ``pip install -r requirements.txt`` to install project back end dependencies.
7. Run ``python manage.py runserver`` to run the server.
8. You will likely have migrations (i.e., ``You have N unapplied migration(s).``). To fix this, simply run ``python manage.py migrate``. 
9. To create a superuser, run ``python manage.py createsuperuser``. Provide the username, email, and password as prompted by your terminal. This will be the account you use to access the admin panel.

### Post-setup
* **Important: use the ``127.0.0.1`` version of localhost rather than ``localhost`` to ensure cookies are properly handled.** 
* You can access the back end API at [127.0.0.1:8000/api/](http://127.0.0.1:8000/api/)
* You can access the back end admin panel at [127.0.0.1:8000/api/admin/](http://127.0.0.1:8000/api/admin/) and login using the details you entered when prompted earlier from ``python manage.py createsuperuser``.
* You can access the back end documentation at [127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs/)

### Development
* Please write API documentation and testing. No one will know or care but it is important for future developers.
* Do follow best practices.
    * Pillars of code quality: 
        * Make code reuseable
        * Avoid surprises
        * Make code hard to misuse
        * Make code modular 
        * Make code reuseable
        * Make code testable and test it properly
    * Goals to achieve in writing good code: 
        * It should work 
        * It should keep on working
        * It should be adaptable to changing requirements 
        * It should not reinvent the wheel 
