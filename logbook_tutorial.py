from logbook import Logger, StreamHandler
import sys
StreamHandler(sys.stdout).push_application()
log = Logger('My Awesome Logger')
log.warn('This is too cool for stdlib')