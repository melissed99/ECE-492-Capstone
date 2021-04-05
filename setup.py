from setuptools import setup

setup(
    name='ECE-492-Capstone',
    version='',
    packages=['web_app'],
    url='',
    license='',
    author='',
    author_email='',
    description='Smart Office Defender',
    install_requires=[
        'Flask',
    ],
    entry_points={
        'console_scripts': [
            'web-app = web_app.web_app:main'
        ]
    },
    package_data={
        'web_app': [
            'templates/*'
        ]
    }
)
