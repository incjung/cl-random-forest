(ql:quickload :cl-random-forest)
(ql:quickload :lfarm-server)

;; PLEASE run on each server
(lfarm-server:start-server "node1" 11524 :background t) ;; on node1

(lfarm-server:start-server "node2" 11524 :background t) ;; on node2

(lfarm-server:start-server "node3" 11524 :background t) ;; on node3

(lfarm-server:start-server "node4" 11524 :background t) ;; on node4


;; on client 
(ql:quickload :lfarm-client)
(ql:quickload :cl-random-forest)
(defpackage clrf-multinode
  (:use :cl :cl-random-forest))

(in-package :clrf-multinode)

;; Connect to the servers
(setf lfarm:*kernel* (lfarm:make-kernel '(("node1" 11524) ("node2" 11524) ("node3" 11524) ("node4" 11524))))

;; load Library
(lfarm:broadcast-task #'ql:quickload :cl-random-forest)
;; Init random-state
(lfarm:broadcast-task
 (lambda ()
   (setq *random-state* (make-random-state t))))

;; It is necessary to distribute the dataset file to each server beforehand
(lfarm:broadcast-task
 (lambda ()
   (defparameter mnist-dim 784)
   (defparameter mnist-n-class 10)

   (multiple-value-bind (datamat target)
      (cl-random-forest.utils:read-data "/mapr/cluster1/mnist/mnist.scale" mnist-dim)
     (defparameter mnist-datamatrix datamat)
     (defparameter mnist-target target))

   (loop for i from 0 below (length mnist-target) do
     (incf (aref mnist-target i)))))

;; build model
(time
  (lfarm:broadcast-task
   (lambda ()
     ;;multi-thread conf
     (setf lparallel:*kernel* (lparallel:make-kernel 4))

     (defparameter mnist-forest
       (cl-random-forest:make-forest mnist-n-class mnist-datamatrix mnist-target
                 :n-tree 1000 :bagging-ratio 0.1 :max-depth 10 :n-trial 10 :min-region-samples 5)))))

;; Predict
(defun predict-forest-multinode (index)
  (let* ((result (lfarm:broadcast-task
                  `(lambda ()
                     (clrf::class-distribution-forest mnist-forest mnist-datamatrix ,index))))
         (n-node (length result))
         (n-dim (length (aref result 0)))
         (class-dist (make-array n-dim :element-type 'double-float :initial-element 0d0)))
    (print result)
    (loop for res across result do
      (loop for i from 0 below n-dim
            for elem across res do
              (incf (aref class-dist i) elem)))
    (loop for i from 0 below n-dim do
      (setf (aref class-dist i) (/ (aref class-dist i) n-node)))
    class-dist))

(predict-forest-multinode 0)
(predict-forest-multinode 1)

