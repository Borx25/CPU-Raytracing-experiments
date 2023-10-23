# CPU-Raytracing-experiments

## Backward path tracer:
- Paths start when a ray (direction + origin + payload) is created at the camera.
- The payload is the throughput, that is, what light can that path carry back to the camera.
- Rays are intersected with the scene.
- Intersections can terminate or extend a path with an additional ray and will reduce throughput according to the Bidirectional reflectance distribution function (BRDF).
    This is because some light is absorbed or otherwise scattered in a direction other than the one we are evaluating.
    The BRDF tells us for some radiance R coming to surface from direction A how much would reflect in direction B.
    Applying this to a backward path tracing intersection: A is the extension ray, that is the direction that we will look for light next and B is our current ray.
    The question is: for light coming from there (extension) how much (throughput) would reflect exactly through the path back to the camera?
- When a path is terminated at a light source (could be the sky) we now it can carry the radiance from that light
  multiplied by the throughput of the path back to the camera, regardless of how many bounces it took to get there.
- Multiple samples would be evaluated per pixel, each generating different camera rays to simulate depth of field, different extension rays for materials that arent a perfect mirror, etc [Monte carlo integration]

  With additional complications detailed below this is what Renderer::Accummulate does. non-recursive pseudocode overview:
```
  for pixel in image {
    path.ray = camera.generate_ray(pixel.x, pixel.y)
    path.radiance = 0.0f
    path.throughput = vec3{1, 1, 1} //full throughput at the start

    for bounce in max_bounces{
      hit = scene.intersect(path.ray) //obtain intersection point, normal vector, distance from ray origin, material...
      if hit {
        material = get_material(hit)
        extension_dir = material.generate_extension_ray(path.ray.direction, hit.N)
        path.radiance += material.emission * throughput //light emitted by the object we hit
        path.throughput *= material.BRDF(path.ray, extension_ray) * dot(extension_dir, hit.N)
        path.ray = extension_ray //update direction for the next bounce
        path.origin = hit.position //next ray originates at the hit position      
      } else { //miss
        path.radiance += sky_radiance(path.ray) * throughput //light radiated by the sky
        break //end path
      }
    }
    out[pixel] += path.radiance //accumulate the radiance from this sample
  }
```    

## Bidirectional reflectance distribution function: [WIP]
  - Only lambertian is implemented at the moment

## Additional complications implemented: [WIP]
  - Bounding volume hierarchy for faster traversal of the scene
  - SIMD sphere intersection
  - Light sampling and multiple importance sampling
  - Tracing and shading a stream of rays instead of individual rays
  - Median of means instead of mean as the estimator to reduce fireflies
  
