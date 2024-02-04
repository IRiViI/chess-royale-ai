import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chessformer",
    version="1.0.0",
    author="Rick Vink",
    author_email="rckvnk@gmail.com",
    description="AI for the game chess royale",
    url="https://github.com/IRiViI/mcts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    keywords=["ai", "deeplearning", "chess", "chess royale"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy"],
)