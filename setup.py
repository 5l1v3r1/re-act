from setuptools import setup

setup(
    name='re-act',
    version='0.0.1',
    description='Consistent exploration via live SGD',
    url='https://github.com/unixpickle/re-act',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ai reinforcement learning',
    packages=['re_act'],
    install_requires=[
        'anyrl>=0.12.0,<0.13.0',
        'gym>=0.9.6,<0.11.0',
        'mazenv>=0.4.0,<0.5.0',
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
    },
)
