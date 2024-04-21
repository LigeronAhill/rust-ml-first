#[derive(Debug)]
pub enum Error {
    InvalidNumberOfInputs,
    InvalidTargetLength,
    MatrixError(matrix::Error),
    IoError(std::io::Error),
    SerdeError(serde_json::Error)
}
pub type Result<T> = core::result::Result<T, Error>;

impl From<matrix::Error> for Error {
    fn from(value: matrix::Error) -> Self {
        Self::MatrixError(value)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}
impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::SerdeError(value)
    }
}